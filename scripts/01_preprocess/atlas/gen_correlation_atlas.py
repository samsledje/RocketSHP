import os
import time

import cuarray
import mdtraj as md
import netcalc
import netchem
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from rocketshp import config

os.environ["LOGURU_LEVEL"] = "INFO"

ATLAS_DATA_DIR = config.RAW_DATA_DIR / "atlas"
ATLAS_PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR / "atlas"

pdb_files = list(ATLAS_DATA_DIR.glob("*/*.pdb"))
N_REPS = 3
LOCAL_DIST_CUTOFF = 0.75  # in nm
STEP = 10
DO_LOCAL_ALIGN = True

for pdb_f in tqdm(pdb_files, total=len(pdb_files)):
    pdb_code = pdb_f.stem

    for rep in range(1, N_REPS + 1):
        xtc_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_prod_R{rep}_fit.xtc"
        if not xtc_f.exists():
            logger.warning(f"Missing {xtc_f}")
            continue

        logger.info(f"Processing {pdb_code} rep {rep}")
        logger.info("Loading pdb structure")
        traj = md.load_pdb(pdb_f)
        logger.info("Loading MD trajectory")
        traj = md.load(str(xtc_f), top=pdb_f)
        logger.info("Superposing trajectory")
        traj.superpose(traj, 0)
        traj.center_coordinates()
        logger.info("Slicing trajectory")
        traj = traj[::STEP]
        atoms_to_keep = [a.index for a in traj.topology.atoms if a.name == "CA"]
        traj.restrict_atoms(atoms_to_keep)

        dcd_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_prod_R{rep}_fit.dcd"
        pdb_ca_f = ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_prod_R{rep}_fit.ca.pdb"
        logger.info(f"Writing {traj} to {str(dcd_f)}, {str(pdb_ca_f)}")
        traj.save_dcd(str(dcd_f))
        traj.save_pdb(str(pdb_ca_f))

        starttime = time.time()
        first_frame = 0
        last_frame = len(traj) - 1
        num_frames = last_frame - first_frame + 1

        logger.info("Creating netchem graph")
        graph = netchem.Graph()
        graph.init(
            trajectoryFile=str(dcd_f),
            topologyFile=str(pdb_ca_f),
            firstFrame=first_frame,
            lastFrame=last_frame,
        )

        # Output correlation matrix
        R = cuarray.FloatCuArray()

        # Correlation pairs to compute
        ab = cuarray.IntCuArray()
        num_nodes = graph.numNodes()
        num_node_pairs = num_nodes**2

        # Define all pair correlations that will be computed
        ab.init(num_node_pairs, 2)
        for i in range(num_nodes):
            for j in range(num_nodes):
                node_pair_index = i * num_nodes + j
                ab[node_pair_index][0] = i
                ab[node_pair_index][1] = j

        # Number of data points
        n = graph.numFrames()

        # Dimensionality of the data
        d = 3
        xd = 2

        # K-nearest-neighbors
        k = 6

        # CUDA platform
        platform = 0

        if DO_LOCAL_ALIGN:
            # Locally align nodes
            logger.info("Locally aligning nodes")

            def residue_com(traj, res, frame=0):
                first_frame_coords = traj.xyz[frame, :, :]
                com = np.array([0.0, 0.0, 0.0])
                total_mass = 0.0
                for k, atom1 in enumerate(res.atoms):
                    mass = atom1.element.mass
                    com += mass * first_frame_coords[atom1.index, :]
                    total_mass += mass

                com /= total_mass
                return com

            # globally_aligned_nodes = graph.nodeCoordinates()
            locally_aligned_nodes = np.zeros((num_nodes, 3 * num_frames)).astype(
                np.float32
            )
            for i, res1 in enumerate(traj.topology.residues):
                atom1_coords = residue_com(traj, res1)
                close_atom_indices = []
                for j, res2 in enumerate(traj.topology.residues):
                    if i == j:
                        continue
                    atom2_coords = residue_com(traj, res2)
                    dist = np.linalg.norm(atom2_coords - atom1_coords)
                    if dist <= LOCAL_DIST_CUTOFF:
                        close_atom_indices.append(j)

                traj.superpose(
                    traj,
                    atom_indices=close_atom_indices,
                    ref_atom_indices=close_atom_indices,
                )
                # positions = traj.xyz[:,i,:] - traj.xyz[0,i,:]
                atom1_coords_aligned = residue_com(traj, res1)
                positions = np.zeros((traj.n_frames, 3))
                for L in range(traj.n_frames):
                    positions[L, :] = (
                        residue_com(traj, res1, frame=L) - atom1_coords_aligned
                    )

                locally_aligned_nodes[i, 0:num_frames] = positions[:, 0]
                locally_aligned_nodes[i, num_frames : 2 * num_frames] = positions[:, 1]
                locally_aligned_nodes[i, 2 * num_frames : 3 * num_frames] = positions[
                    :, 2
                ]

            graph.nodeCoordinates().fromNumpy2D(
                locally_aligned_nodes.astype(np.float32)
            )

        local_suff = "local_" if DO_LOCAL_ALIGN else ""

        # Compute generalized correlation and output to proteinG_R
        logger.info(
            "Performing generalized correlation computation "
            f"on {n} data points with {num_nodes} nodes."
        )
        netcalc.generalizedCorrelation(
            X=graph.nodeCoordinates(),
            R=R,
            ab=ab,
            k=k,
            n=n,
            d=d,
            xd=xd,
            platform=platform,
        )
        logger.info(f"Time: {time.time() - starttime:.3f} s.")

        # Gen. Corr. in numpy array object
        logger.info("Saving results")
        R_np = R.toNumpy2D().reshape(num_nodes, num_nodes)
        # corr_matrix_filename = str(ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_{rep}_{local_suff}step{STEP}_corr_matrix.txt")
        corr_matrix_filename = str(
            ATLAS_DATA_DIR
            / pdb_code[:2]
            / f"{pdb_code}_{rep}_{local_suff}step{STEP}_corr_matrix.pt"
        )
        logger.info(f"Saving matrix to: {corr_matrix_filename}")
        # np.savetxt(corr_matrix_filename, R_np)
        torch.save(torch.tensor(R_np), corr_matrix_filename)
        logger.info(f"Total time: {time.time() - starttime:.3f} s.")

        # Plotting
        # logger.info("Plotting")
        # import numpy as np
        # import matplotlib.pyplot as plt

        # R_np = np.flip(R_np, axis=0)
        # R_figure_x = [i for i in range(num_nodes)]
        # R_figure_y = [i for i in range(num_nodes)]

        # im = plt.imshow(R_np, vmin=0.0, vmax=1.0, extent=[1, num_nodes+1, 1, num_nodes+1],
        #                 cmap=plt.cm.jet)
        # im.set_interpolation('bilinear')
        # plt.xlabel("Residue number")
        # plt.ylabel("Residue number")
        # cbar = plt.colorbar(im)
        # plt.savefig(str(ATLAS_DATA_DIR / pdb_code[:2] / f"{pdb_code}_{rep}_{local_suff}step{STEP}_corr_matrix.png"))

        # break
