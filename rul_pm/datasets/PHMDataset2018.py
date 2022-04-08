import gzip
import io
import logging
import os
import pickle
import tarfile
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import gdown
import pandas as pd
from joblib import Memory
from rul_pm import CACHE_PATH, DATASET_PATH
from rul_pm.datasets.lives_dataset import AbstractLivesDataset
from tqdm.auto import tqdm
import shutil
import numpy as np


logger = logging.getLogger(__name__)

memory = Memory(CACHE_PATH, verbose=0)

COMPRESSED_FILE = "phm_data_challenge_2018.tar.gz"
FOLDER = "phm_data_challenge_2018"


URL = "https://drive.google.com/uc?id=15Jx9Scq9FqpIGn8jbAQB_lcHSXvIoPzb"
OUTPUT = COMPRESSED_FILE


def download(path: Path):
    logger.info("Downloading dataset...")
    gdown.download(URL, str(path / OUTPUT), quiet=False)


def prepare_raw_dataset(path: Path):
    def track_progress(members):
        for member in tqdm(members, total=70):
            yield member
    path = path / 'raw'
    path.mkdir(parents=True, exist_ok=True)
    if not (path / OUTPUT).resolve().is_file():
        download(path)
    logger.info("Decompressing  dataset...")
    with tarfile.open(path  / OUTPUT, "r") as tarball:
        tarball.extractall(path=path, members=track_progress(tarball))
    shutil.move(str(path / 'phm_data_challenge_2018' / 'train'),
                str(path / 'train'))
    shutil.move(str(path / 'phm_data_challenge_2018' / 'test'),
                str(path / 'test'))
    shutil.rmtree(str(path / 'phm_data_challenge_2018'))
    (path / OUTPUT).unlink()


class FailureType(Enum):
    FlowCoolPressureDroppedBelowLimit = "FlowCool Pressure Dropped Below Limit"
    FlowcoolPressureTooHighCheckFlowcoolPump = (
        "Flowcool Pressure Too High Check Flowcool Pump"
    )
    FlowcoolLeak = "Flowcool leak"

    @staticmethod
    def that_starth_with(s: str):
        for f in FailureType:
            if s.startswith(f.value):
                return f
        return None


def prepare_dataset(dataset_path: Path):
    (dataset_path / "processed" / "lives").mkdir(exist_ok=True, parents=True)

    files = list(Path(dataset_path / "raw" / "train").resolve().glob("*.csv"))
    faults_files = list(
        Path(dataset_path / "raw" / "train" / "train_faults").resolve().glob("*.csv")
    )
    files = {file.stem[0:6]: file for file in files}
    faults_files = {file.stem[0:6]: file for file in faults_files}
    dataset_data = []
    for filename in tqdm(faults_files.keys(), "Processing files"):
        tool = filename[0:6]
        data_file = files[tool]
        logger.info(f"Loading data file {files[tool]}")
        data = pd.read_csv(data_file).dropna().set_index("time")

        c = data.columns.tolist()
        fault_data_file = faults_files[filename]
        fault_data = pd.read_csv(fault_data_file).set_index("time")
        fault_data['fault_number'] = range(fault_data.shape[0])
        data = pd.merge_asof(data, fault_data, on='time', direction='forward').set_index('time')

        for life_index, life_data in data.groupby('fault_number'):
            if life_data.shape[0] == 0:
                continue
            failure_type = FailureType.that_starth_with(life_data['fault_name'].iloc[0])
            output_filename = f"Life_{int(life_index)}_{tool}_{failure_type.name}.pkl.gzip"
            dataset_data.append(
                (tool, life_data.shape[0], failure_type.value, output_filename)
            )
            life = life_data.copy()
            life['RUL'] = life.index[-1] - life.index 
            with gzip.open(
                dataset_path / "processed" / "lives" / output_filename, "wb"
            ) as file:
                pickle.dump(life_data, file)

    df = pd.DataFrame(
        dataset_data, columns=["Tool", "Number of samples", "Failure Type", "Filename"]
    )
    df.to_csv(dataset_path / "processed" / "lives" / "lives_db.csv")


class PHMDataset2018(AbstractLivesDataset):
    def __init__(
        self,
        failure_types: Union[FailureType, List[FailureType]] = [l for l in FailureType],
        tools: Union[str, List[str]] = "all",
        path: Path = DATASET_PATH,
    ):
        super().__init__()
        if not isinstance(failure_types, list):
            failure_types = [failure_types]
        self.failure_types = failure_types

        if isinstance(tools, str) and tools != "all":
            tools = [tools]
        self.tools = tools

        self.path = path
        self.dataset_path = path / FOLDER

        self.procesed_path = self.dataset_path / "processed" / "lives"
        self.lives_table_filename = self.procesed_path / "lives_db.csv"
        if not self.lives_table_filename.is_file():
            if not (self.dataset_path / "raw" / "train").is_dir():
                prepare_raw_dataset(self.dataset_path)
            prepare_dataset(self.dataset_path)

        self.lives = pd.read_csv(self.lives_table_filename)
        if tools != "all":
            self.lives = self.lives[self.lives["Tool"].isin(self.tools)]
        self.lives = self.lives[
            self.lives["Failure Type"].isin([a.value for a in self.failure_types])
            
        ]
        valid = []
        for i, (j, r) in enumerate(self.lives.iterrows()):
            df = self._load_life(r['Filename'])
            df = df[df['FIXTURESHUTTERPOSITION'] == 1]
            if df.shape[0] > 0:
                valid.append(i)
        self.lives =  self.lives.iloc[valid, :]

    @property
    def n_time_series(self) -> int:
        return self.lives.shape[0]

    def _load_life(self, filename:str)->pd.DataFrame:
        with gzip.open(
            self.procesed_path / filename, "rb"
        ) as file:
            df = pickle.load(file)
        return df

    def get_time_series(self, i: int) -> pd.DataFrame:
        """
        Paramters
        ---------
        i:int


        Returns
        -------
        pd.DataFrame
            DataFrame with the data of the life i
        """
        df = self._load_life(self.lives.iloc[i]["Filename"])

        df = df[df['FIXTURESHUTTERPOSITION'] == 1].copy().dropna()
        df = df[df['ETCHAUXSOURCETIMER'].diff() != 0]
        df = df[df['ETCHSOURCEUSAGE'].diff() != 0]


        df['RUL'] = np.array(list(range(df.shape[0]-1, -1, -1))) * 4
        
            
        return df

    @property
    def rul_column(self) -> str:
        return 'RUL'
