"""Run DRL-VNE on a fixed five-node substrate and two virtual requests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

import graph_u
from drl_vne import (
    DEFAULT_DRL_CHECKPOINT,
    DRLVNEEnvironment,
    load_policy,
    undirected_edges,
)

SUBSTRATE_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2))
NODE_DOMAINS = ("space", "space", "air", "ground", "ground")


def _make_graph(
    nodes: int,
    undirected_links: tuple[tuple[int, int], ...],
    cpu_demands: tuple[float, ...],
    bandwidths: tuple[float, ...],
    delays: tuple[float, ...],
) -> graph_u.Graph:
    if len(cpu_demands) != nodes:
        raise ValueError(f"Expected {nodes} CPU values, received {len(cpu_demands)}.")
    if not (len(undirected_links) == len(bandwidths) == len(delays)):
        raise ValueError("Each graph link must have exactly one bandwidth and delay value.")

    directed_links = {
        (str(left), str(right))
        for edge in undirected_links
        for left, right in (edge, edge[::-1])
    }
    parameters = graph_u.Parameters(1, 2, 1, 2, 0, 100, 0, 100, 1, 2, 1, 2, 1, 2)
    graph = graph_u.Graph(nodes, directed_links, parameters)
    graph.node_weights.update({node: value for node, value in enumerate(cpu_demands)})
    for (left, right), bandwidth, delay in zip(undirected_links, bandwidths, delays):
        for source, target in ((left, right), (right, left)):
            graph.edge_weights[(str(source), str(target))] = bandwidth
            graph.delay[(str(source), str(target))] = delay
    return graph


def _make_substrate() -> graph_u.Graph:
    substrate = _make_graph(
        nodes=5,
        undirected_links=SUBSTRATE_EDGES,
        cpu_demands=(200, 180, 220, 170, 210),
        bandwidths=(80, 95, 70, 110, 60, 100),
        delays=(2, 2, 3, 2, 3, 1),
    )
    substrate.node_domains = {
        node: domain for node, domain in enumerate(NODE_DOMAINS)
    }
    substrate.topology_name = "DRL-VNE-5-node-test"
    substrate.use_delay_routing = True
    return substrate


def _make_requests() -> list[graph_u.Graph]:
    return [
        _make_graph(
            nodes=3,
            undirected_links=((0, 1), (1, 2)),
            cpu_demands=(40, 60, 40),
            bandwidths=(20, 20),
            delays=(1, 1),
        ),
        _make_graph(
            nodes=2,
            undirected_links=((0, 1),),
            cpu_demands=(35, 30),
            bandwidths=(25,),
            delays=(1,),
        ),
    ]


def _unique_bandwidth(graph: Any) -> dict[tuple[int, int], float]:
    return {
        edge: float(graph.edge_weights[(str(edge[0]), str(edge[1]))])
        for edge in undirected_edges(graph)
    }


def _build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("drl_vne_test")
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
    checkpoint: str | Path = DEFAULT_DRL_CHECKPOINT,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    log_file = (
        Path(log_path)
        if log_path is not None
        else Path(__file__).with_name("drl_vne_test.log")
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = _build_logger(log_file)
    try:
        torch.manual_seed(7)
        substrate = _make_substrate()
        requests = _make_requests()
        policy = load_policy(checkpoint)
        environment = DRLVNEEnvironment(substrate)

        logger.info("Starting the DRL-VNE 5-node embedding test.")
        logger.info("Checkpoint: %s", Path(checkpoint).resolve())
        logger.info(
            "Substrate: %d nodes, %d undirected links",
            substrate.nodes,
            len(SUBSTRATE_EDGES),
        )
        for node in range(substrate.nodes):
            logger.info(
                "Substrate node %s: domain=%s CPU=%g",
                "ABCDE"[node],
                substrate.node_domains[node],
                substrate.node_weights[node],
            )
        for (left, right), capacity in _unique_bandwidth(substrate).items():
            logger.info(
                "Substrate link %s-%s: bandwidth=%g delay=%g",
                "ABCDE"[left],
                "ABCDE"[right],
                capacity,
                substrate.delay[(str(left), str(right))],
            )

        results = []
        with torch.no_grad():
            for request_number, request in enumerate(requests, start=1):
                cpu_before = dict(environment.substrate.node_weights)
                bandwidth_before = _unique_bandwidth(environment.substrate)
                request_edges = DRLVNEEnvironment.virtual_edges(request)
                logger.info(
                    "Request %d begins: %d virtual nodes, %d virtual links",
                    request_number,
                    request.nodes,
                    len(request_edges),
                )
                for node, demand in request.node_weights.items():
                    logger.info(
                        "Request %d virtual node %s: CPU demand=%g",
                        request_number,
                        "abc"[node],
                        demand,
                    )
                for left, right, demand in request_edges:
                    logger.info(
                        "Request %d virtual link %s-%s: bandwidth demand=%g",
                        request_number,
                        "abc"[left],
                        "abc"[right],
                        demand,
                    )

                success, info, _, _ = environment.embed_request(
                    request, policy, sample=False
                )
                logger.info(
                    "Request %d result: %s",
                    request_number,
                    "ACCEPTED" if success else f"REJECTED ({info.get('reason')})",
                )
                for virtual_node, substrate_node in info.get("placement", {}).items():
                    logger.info(
                        "Request %d node mapping: %s -> %s "
                        "(substrate CPU %g -> %g; demand=%g)",
                        request_number,
                        "abc"[virtual_node],
                        "ABCDE"[substrate_node],
                        cpu_before[substrate_node],
                        environment.substrate.node_weights[substrate_node],
                        request.node_weights[virtual_node],
                    )
                for route in info.get("routes", []):
                    substrate_path = "-".join(
                        "ABCDE"[node] for node in route["substrate_path"]
                    )
                    logger.info(
                        "Request %d link mapping: virtual link %s-%s -> substrate path %s "
                        "(bandwidth demand=%g)",
                        request_number,
                        "abc"[route["virtual_edge"][0]],
                        "abc"[route["virtual_edge"][1]],
                        substrate_path,
                        route["bandwidth_demand"],
                    )
                    for left, right in zip(
                        route["substrate_path"],
                        route["substrate_path"][1:],
                    ):
                        edge = tuple(sorted((left, right)))
                        bandwidth_after = _unique_bandwidth(environment.substrate)[edge]
                        logger.info(
                            "Request %d link resource: %s-%s bandwidth %g -> %g",
                            request_number,
                            "ABCDE"[edge[0]],
                            "ABCDE"[edge[1]],
                            bandwidth_before[edge],
                            bandwidth_after,
                        )
                if success:
                    logger.info(
                        "Request %d metrics: revenue=%g node_cost=%g edge_cost=%g "
                        "total_cost=%g hops=%d delay=%g revenue_to_cost=%.2f%%",
                        request_number,
                        info["revenue"],
                        info["node_cost"],
                        info["edge_cost"],
                        info["total_cost"],
                        info["hops"],
                        info["total_delay"],
                        100 * info["revenue"] / info["total_cost"]
                        if info["total_cost"]
                        else 0.0,
                    )
                results.append({"accepted": success, **info})

        accepted = sum(result["accepted"] for result in results)
        logger.info(
            "Final result: %d/%d requests embedded",
            accepted,
            len(requests),
        )
        logger.info("Final substrate CPU: %s", environment.substrate.node_weights)
        logger.info("Final substrate link bandwidth: %s", _unique_bandwidth(environment.substrate))
        if accepted != len(requests):
            raise RuntimeError(
                f"Expected both test requests to embed, but only "
                f"{accepted}/{len(requests)} succeeded."
            )
        logger.info("Test passed: both requests embedded by DRL-VNE.")
        return {
            "accepted": accepted,
            "total_requests": len(requests),
            "results": results,
            "log_file": log_file,
        }
    except Exception:
        logger.exception("DRL-VNE test failed.")
        raise
    finally:
        for handler in logger.handlers[:]:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DRL_CHECKPOINT)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(__file__).with_name("drl_vne_test.log"),
    )
    arguments = parser.parse_args()
    run_test(arguments.checkpoint, arguments.log_file)


if __name__ == "__main__":
    main()
