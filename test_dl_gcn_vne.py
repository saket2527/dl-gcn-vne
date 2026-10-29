"""Run DL-GCN-VNE on a fixed five-node substrate and two VNRs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

import graph_u
from dl_vne import DEFAULT_CHECKPOINT, VNEEnvironment, load_policy

SUBSTRATE_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2))
NODE_DOMAINS = ("space", "space", "air", "ground", "ground")


def _make_graph(
    nodes: int,
    undirected_edges: tuple[tuple[int, int], ...],
    cpu: tuple[float, ...],
    bandwidth: tuple[float, ...],
    delays: tuple[float, ...],
) -> graph_u.Graph:
    directed_edges = {
        (str(left), str(right))
        for edge in undirected_edges
        for left, right in (edge, edge[::-1])
    }
    parameters = graph_u.Parameters(1, 2, 1, 2, 0, 100, 0, 100, 1, 2, 1, 2, 1, 2)
    graph = graph_u.Graph(nodes, directed_edges, parameters)
    graph.node_weights.update({node: value for node, value in enumerate(cpu)})
    graph.node_domains = {}
    for node, domain in enumerate(NODE_DOMAINS[:nodes]):
        graph.node_domains[node] = domain
    for (left, right), capacity, delay in zip(undirected_edges, bandwidth, delays):
        for source, target in ((left, right), (right, left)):
            graph.edge_weights[(str(source), str(target))] = capacity
            graph.delay[(str(source), str(target))] = delay
    return graph


def _make_requests() -> list[graph_u.Graph]:
    requests = [
        _make_graph(
            nodes=3,
            undirected_edges=((0, 1), (1, 2)),
            cpu=(40, 60, 40),
            bandwidth=(20, 20),
            delays=(1, 1),
        ),
        _make_graph(
            nodes=2,
            undirected_edges=((0, 1),),
            cpu=(35, 30),
            bandwidth=(25,),
            delays=(1,),
        ),
    ]
    return requests


def _unique_bandwidth(graph: Any) -> dict[tuple[int, int], float]:
    return {
        tuple(sorted((int(left), int(right)))): float(value)
        for (left, right), value in graph.edge_weights.items()
        if int(left) < int(right)
    }


def _build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("dl_gcn_vne_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def run_test(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    log_file = Path(log_path) if log_path is not None else Path(__file__).with_name("dl_gcn_vne_test.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = _build_logger(log_file)
    try:
        torch.manual_seed(7)
        substrate = _make_graph(
            nodes=5,
            undirected_edges=SUBSTRATE_EDGES,
            cpu=(200, 180, 220, 170, 210),
            bandwidth=(80, 95, 70, 110, 60, 100),
            delays=(2, 2, 3, 2, 3, 1),
        )
        substrate.topology_name = "DL-GCN-VNE-5-node-test"
        requests = _make_requests()
        policy = load_policy(checkpoint, device="cpu")
        if policy.feature_dim != 7:
            raise ValueError(
                f"The test substrate provides 7 features, but the checkpoint expects "
                f"{policy.feature_dim}."
            )
        environment = VNEEnvironment(substrate, device="cpu")

        logger.info("Starting DL-GCN-VNE inference test; no other embedding algorithm is used.")
        logger.info("Checkpoint: %s", Path(checkpoint).resolve())
        logger.info("Substrate: %d nodes, %d undirected links", substrate.nodes, len(SUBSTRATE_EDGES))
        for node in range(substrate.nodes):
            logger.info(
                "Substrate node %d: domain=%s CPU=%g",
                node, substrate.node_domains[node], substrate.node_weights[node],
            )
        for edge, capacity in _unique_bandwidth(substrate).items():
            logger.info(
                "Substrate link %s: bandwidth=%g delay=%g",
                edge, capacity, substrate.delay[(str(edge[0]), str(edge[1]))],
            )

        results = []
        for request_number, request in enumerate(requests, start=1):
            cpu_before = dict(environment.substrate.node_weights)
            bandwidth_before = _unique_bandwidth(environment.substrate)
            logger.info(
                "Request %d: begin embedding (%d virtual nodes, %d virtual links)",
                request_number, request.nodes, len(_unique_bandwidth(request)),
            )
            for node, demand in request.node_weights.items():
                logger.info("Request %d virtual node %d: CPU demand=%g", request_number, node, demand)
            for edge, demand in _unique_bandwidth(request).items():
                logger.info("Request %d virtual link %s: bandwidth demand=%g", request_number, edge, demand)

            success, _, info = environment.embed_request(request, policy, sample=False)
            logger.info(
                "Request %d result: %s",
                request_number,
                "ACCEPTED" if success else f"REJECTED ({info.get('reason', 'unknown reason')})",
            )
            for virtual_node, substrate_node in info.get("placement", {}).items():
                logger.info(
                    "Request %d placement: virtual node %d -> substrate node %d "
                    "(CPU %g -> %g; demand=%g)",
                    request_number,
                    virtual_node,
                    substrate_node,
                    cpu_before[substrate_node],
                    environment.substrate.node_weights[substrate_node],
                    request.node_weights[virtual_node],
                )
            for route in info.get("route_paths", []):
                logger.info(
                    "Request %d routing: virtual link %s -> substrate path %s "
                    "(bandwidth demand=%g)",
                    request_number,
                    route["virtual_edge"],
                    route["substrate_path"],
                    route["bandwidth_demand"],
                )
                for left, right in zip(route["substrate_path"], route["substrate_path"][1:]):
                    edge = tuple(sorted((left, right)))
                    logger.info(
                        "Request %d link resource: substrate link %s bandwidth %g -> %g",
                        request_number,
                        edge,
                        bandwidth_before[edge],
                        _unique_bandwidth(environment.substrate)[edge],
                    )
            if success:
                logger.info(
                    "Request %d metrics: revenue=%g node_cost=%g edge_cost=%g hops=%d delay=%g",
                    request_number,
                    info["revenue"],
                    info["node_cost"],
                    info["edge_cost"],
                    info["hops"],
                    info["total_delay"],
                )
            results.append({"accepted": success, **info})

        accepted = sum(result["accepted"] for result in results)
        logger.info("Final result: %d/%d virtual network requests embedded", accepted, len(requests))
        logger.info("Final substrate CPU: %s", environment.substrate.node_weights)
        logger.info("Final substrate link bandwidth: %s", _unique_bandwidth(environment.substrate))
        if accepted != len(requests):
            raise RuntimeError(f"Expected both requests to embed, but only {accepted}/{len(requests)} succeeded.")
        logger.info("Test passed: both VNRs were embedded by DL-GCN-VNE.")
        return {"accepted": accepted, "total_requests": len(requests), "results": results, "log_file": log_file}
    except Exception:
        logger.exception("DL-GCN-VNE test failed.")
        raise
    finally:
        for handler in logger.handlers[:]:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(__file__).with_name("dl_gcn_vne_test.log"),
    )
    arguments = parser.parse_args()
    run_test(arguments.checkpoint, arguments.log_file)


if __name__ == "__main__":
    main()
