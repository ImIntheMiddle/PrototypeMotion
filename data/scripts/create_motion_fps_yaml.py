import os
from pathlib import Path
from typing import Optional
import tqdm
import numpy as np
import typer
import yaml
import glob

SMPL_DATASETS_NAME_DICT = {"ACCAD": "ACCAD", "BMLmovi": "BMLmovi", "BioMotionLab_NTroje": "BMLrub", "CMU": "CMU", "DFaust_67": "DFaust", "EKUT": "EKUT", "Eyes_Japan_Dataset": "Eyes_Japan_Dataset", "HumanEva": "HumanEva", "Transitions_mocap": "Transitions", "MPI_HDM05": "HDM05", "MPI_mosh": "MoSh", "MPI_Limits": "PosePrior", "KIT": "KIT", "SFU": "SFU", "SSM_synced": "SSM", "TotalCapture": "TotalCapture"}

def main(
    main_motion_dir: Path,
    humanoid_type: str = "smpl",
    amass_fps_file: Optional[Path] = "data/yaml_files/motion_fps_amass.yaml",
    output_path: Optional[Path] = "data/yaml_files",
):
    if humanoid_type == "smplx":
        assert (
            amass_fps_file is not None
        ), "Please provide the amass_fps_file since amass-x fps is wrong."
        amass_fps = yaml.load(open(amass_fps_file, "r"), Loader=yaml.SafeLoader)

    # iterate over folder and all sub folders recursively.
    # load each file.
    # store the full filename in a dictionary.
    # store the entry "motion_fps" in the dictionary.
    # save the dictionary to a yaml file.
    motion_fps_dict = {}
    dir_list = glob.glob(str(main_motion_dir) + "/**", recursive=True)
    dir_num = len([d for d in dir_list if os.path.isdir(d)])
    bar = tqdm.tqdm(os.walk(main_motion_dir), total=dir_num, desc="Processing files")
    for root, dirs, files in bar:
        # Ignore folders with name "-retarget" or "-smpl" or "-smplx"
        if "-retarget" in root or "-smpl" in root or "-smplx" in root:
            continue
        if humanoid_type == "smpl":
            if root == str(main_motion_dir):
                continue
            if root.split(str(main_motion_dir))[-1].split("/")[1] not in SMPL_DATASETS_NAME_DICT.keys():
                continue
        bar.set_description(f"Processing {root}, out of {dir_num} files")
        for file in files:
            if (
                file.endswith(".npz")
                and file != "shape.npz"
                and "stagei.npz" not in file
            ):
                # remove the main_motion_dir from the root
                save_root = root.replace(str(main_motion_dir), "")
                # remove any leading slashes
                save_root = save_root.lstrip("/")

                file_rename = (
                    save_root
                    + "/"
                    + file.replace(".npz", ".npy")
                    .replace("-", "_")
                    .replace(" ", "_")
                    .replace("(", "_")
                    .replace(")", "_")
                )

                if humanoid_type == "smplx":
                    amass_filename = file_rename.replace("_stageii", "_poses")
                    amass_filename = amass_filename.replace("SSM/", "SSM_synced/")
                    amass_filename = amass_filename.replace("HDM05/", "MPI_HDM05/")
                    amass_filename = amass_filename.replace("MoSh/", "MPI_mosh/")
                    amass_filename = amass_filename.replace("PosePrior/", "MPI_Limits/")
                    amass_filename = amass_filename.replace(
                        "TCDHands/", "TCD_handMocap/"
                    )
                    amass_filename = amass_filename.replace(
                        "Transitions/", "Transitions_mocap/"
                    )
                    amass_filename = amass_filename.replace("DFaust/", "DFaust_67/")
                    amass_filename = amass_filename.replace(
                        "BMLrub/", "BioMotionLab_NTroje/"
                    )

                    if amass_filename in amass_fps:
                        framerate = amass_fps[amass_filename]
                    else:
                        if "TotalCapture" in file_rename or "SSM" in file_rename:
                            framerate = 60
                        elif "KIT" in file_rename:
                            framerate = 100
                        elif "BMLmovi" in file_rename:
                            framerate = 120
                        # motion_data = dict(
                        #     np.load(open(root + "/" + file, "rb"), allow_pickle=True)
                        # )
                        # elif "mocap_frame_rate" in motion_data:
                        #     framerate = motion_data["mocap_frame_rate"]
                        else:
                            print(f"{file_rename} has no framerate")
                            continue
                else:
                    if "smplx" in file_rename:
                        continue
                    motion_data = dict(
                        np.load(open(root + "/" + file, "rb"), allow_pickle=True)
                    )
                    if "mocap_framerate" in motion_data:
                        framerate = motion_data["mocap_framerate"]
                    else:
                        raise Exception(f"{file_rename} has no framerate")

                motion_fps_dict[file_rename] = int(framerate)

    if output_path is None:
        output_path = Path.cwd()
    with open(output_path / f"motion_fps_{humanoid_type}.yaml", "w") as f:
        yaml.dump(motion_fps_dict, f)


if __name__ == "__main__":
    typer.run(main)
