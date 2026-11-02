# NORD

## Execution Environment:

Operation System: Microsoft Windows 10, 64bit.<br />
Physical Memory (RAM) 16.0 GB.<br />


### Prerequisites

Python 3.9.<br />
PyCharm Community Edition 2021.2. <br />
Alib utility for VNE simulation.<br />
Introduction about VNE prblem can be found in below link:<br />
https://www.youtube.com/watch?v=JKB3aVyCMuo&t=506s<br />

### Installation

###  Download the ALIB_Utility tool unzip and copy it to the execution drive.<br /> 

- Configure the alib by following the steps mentioned in the github reposotory [1]. <br />

- Generate the input.pickle file and save it in the P3_ALIB_MASTER\input path. <br />

- Make sure "P3_ALIB_MASTER\input" path contain senario_RedBestel.pickle. If not, generate the substrate network scenario for "senario_RedBestel.pickle" in folder P3_ALIB_MASTER\input and this pickle file contains substrate network information.<br />

###   Download  NORD and keep it in the drive where P3_ALIB_MASTER is present. The  NORD  file contains all executable files related to the proposed and baseline approaches. <br />

- Topsis.py -> The Main file related to the  Proposed NORD approach.<br />
- Greedy.py -> The Main file related to the  Greedy baseline approach.<br />
- Nrm.py -> The Main file related to the  NRM baseline approach [2]. <br /> 
- Rethinking.py -> The Main file related to the  DPGA baseline approach [3]. <br />
- NodeRankCal.py -> The Main file related to the  VNE-NTANRC-D baseline approach [4]. <br />
- PageRank.py -> The Main file related to the  VNE-NTANRC-S baseline approach [4]. <br />
- Puvnp.py -> The Main file related to the  PUVNP  baseline approach [5]. <br />
- Rematch.py -> The Main file related to the  ReMatch  baseline approach [6]. <br />




## Usage

### DRL-VNE

`drl_vne.py` implements the five-stage policy described in *Space-Air-
Ground Integrated Multi-domain Network Resource Orchestration based on Virtual
Network Architecture: a DRL Method*. It trains on the existing
`1_sagin.pickle` topology, evaluates generated VNRs, and adds a
`DRL-VNE` row to `Results.xlsx`:

```bash
python3 drl_vne.py
```

By default, it trains with 1,000 requests in batches of 100, evaluates five
requests, and saves the policy to `drl_vne.pth`. Use
`--training-requests`, `--batch-size`, and `--evaluation-requests` to change
those counts. Add `--skip-training` to reuse an existing checkpoint. The
regular `automate.py` run evaluates this policy on the same VNR batch as
the other enabled algorithms; train the paper policy first.

The policy uses the paper's CPU, adjacent bandwidth, adjacent delay, and
average-hop-distance node features, followed by affine scoring, softmax,
feasibility filtering, and probability-ranked output. Link mapping uses BFS
over residual-bandwidth-feasible links. This repository's virtual delay
demands and SAGIN link delays use incompatible value ranges, so link delay is
used as a feature and equal-hop path tie-breaker rather than a hard delay
constraint. To support the requested revenue-to-cost objective, the final
candidate ranking also discounts placements that require more bandwidth-
weighted hops to already-mapped virtual neighbors; this is an explicit
implementation choice beyond the paper's underspecified output ordering.

Run the interactive embedding test with:

```bash
python3 test_drl_vne.py
```

Enter the substrate node and link counts, then provide the two 1-based node
endpoints for each link. Next, enter the request count and each request's node
and link counts and endpoints. Resource capacities and demands are generated
using the test's built-in ranges. The script uses `drl_vne.pth` and writes each
node mapping, link route, and resource update to `drl_vne_test.log`.

### DL-GCN-VNE

`dl_vne.py` implements the PPO-trained graph-convolutional placement policy.
The trained checkpoint is `gcn_drl_vne.pth`; evaluate it on the configured
workload with:

```bash
python3 dl_vne.py
```

Train or replace the checkpoint with `python3 train_drl.py`. The trainer accepts
`--episodes`, `--requests-per-episode`, and `--checkpoint` to configure training.
The regular `automate.py` run includes DL-GCN-VNE and records its results in
`Results.xlsx`. Run its focused embedding check with:

```bash
python3 test_dl_gcn_vne.py
```

### GAT-VNE

`gat_vne.py` uses a multi-head Graph Attention Network to score feasible
substrate placements. It shares the VNE environment and PPO training objective
with DL-GCN-VNE, but learns attention weights over neighboring substrate
nodes. The trained checkpoint is included as `gat_vne.pth`, so no training is
needed to evaluate it:

```bash
python3 gat_vne.py
```

Run `python3 automate.py` to record a `GAT-VNE` row in `Results.xlsx` alongside
the other enabled algorithms. To retrain or replace the checkpoint, run
`python3 train_gat.py`; options include `--episodes`,
`--requests-per-episode`, `--heads`, and `--checkpoint`.

###  In vne_u.py, we can set the various parameters related to Virtual network requests(VNRs).<br />

- We can set the minimum and maximum number of VMs of VNRs in the create_vne function.<br />

- We can set the virtual network request demands like BandWidth(min, max), CRB(min, max), LocationX(min, max), LocationY(min, max), and Delay(min, max) in vne. append function. <br />
- Example: (1, 5, 1, 10, 0, 100, 0, 100, 1, 4)<br />

- Run vne_u.py after doing any modifications. <br />

### Batch results

`automate.py` runs the enabled algorithms on the SAGIN topology in
`1_sagin.pickle`. Final embedding results are captured in `Results.xlsx`.

### References
[1] E. D. Matthias Rost, Alexander Elvers, “Alib,” https://github.com/vnep-approx/alib, 2020. <br />
[2] P. Zhang, H. Yao, Y. Liu, Virtual network embedding based on computing, network, and storage resource constraints, IEEE Internet of Things Journal 5 (5) (2017) 3298–3304. doi: https://doi.org/10.1109/JIOT.2017.2726120. <br />
[3] Nguyen, Khoa TD, Qiao Lu, and Changcheng Huang. "Rethinking virtual link mapping in network virtualization." In 2020 IEEE 92nd Vehicular Technology Conference (VTC2020-Fall), pp. 1-5. IEEE, 2020, https://ieeexplore.ieee.org/document/9348799. <br />
[4] H. Cao, L. Yang, H. Zhu, Novel node-ranking approach and multiple topology attributes-based embedding algorithm for single-domain virtual network embedding, IEEE Internet of Things Journal 5 (1) (2018) 108–120. doi: https://doi.org/10.1109/JIOT.2017.2773489. <br />
[5] P. Zhang, Incorporating energy and load balance into virtual network embedding process, Computer Communications 129 (2018) 80–88. doi: 
https://doi.org/10.1016/j.comcom.2018.07.027. <br />
[6] A. Satpathy, M. N. Sahoo, L. Behera, C. Swain, ReMatch: An Efficient Virtual Data Center Re-Matching Strategy Based on Matching Theory,
IEEE Transactions on Services Computing (2022). doi: https://doi.org/10.1109/TSC.2022.3183259. <br />

## Contributors
- Mr. Keerthan Kumar T G<br />
https://scholar.google.com/citations?user=fW7bzK8AAAAJ&hl=en <br />
- Mr. anurag satpathy<br />
https://anuragsatpathy.github.io/<br />
- Dr. Sourav kanti addya<br />
https://souravkaddya.in/<br />
- Dr Shashidhar G Koolagudi <br />
https://scholar.google.co.in/citations?user=WAyKHHwAAAAJ&hl=en <br />

## Citation & Acknowledgements
This codebase is available publicly for only for non-commercial use. If you use this code or any content in your research or wish to refer to the baseline models published in the README, please cite the work as:
@article{tg2023nord,<br />
  title={NORD: NOde Ranking-based efficient virtual network embedding over single Domain substrate networks},<br />
  author={TG, Keerthan Kumar and Addya, Sourav Kanti and Satpathy, Anurag and Koolagudi, Shashidhar G},<br />
  journal={Computer Networks},<br />
  volume={225},<br />
  pages={109661},<br />
  year={2023},<br />
  publisher={Elsevier}<br />
}


## Paper Links to Cite
- https://www.sciencedirect.com/science/article/pii/S1389128623001068
- https://ieeexplore.ieee.org/document/10041377


## Contact
If you have any questions, simply write a mail to  Keerthanswamy(AT)gmail(DOT)com.
