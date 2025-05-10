import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from .utils import setup_seed
import datetime
import math

from pathlib import Path

local_path = Path(__file__).parent


class TimeSeriesDataset(Dataset):

    def __init__(self, trajs, time_steps, observe_steps, avg_terms,
                 hist_feature, fcst_feature, avail_fcst_feature):
        
        self.observe_steps = observe_steps

        if len(time_steps.shape) < 2:
            time_steps = time_steps.unsqueeze(0)

        # seperate data into different components
        observed_data = trajs[:, :observe_steps, hist_feature]
        data_to_predict = trajs[:, observe_steps:, fcst_feature]
        tp_to_predict = time_steps[:, observe_steps:]
        observed_tp = time_steps[:, :observe_steps]

        # aggregated if needed
        if avg_terms > 1:
            tp_to_predict = tp_to_predict[..., None]
            observed_tp = observed_tp[..., None]

            observed_data = observed_data.transpose(1, 2)
            observed_data = torch.nn.functional.avg_pool1d(
                observed_data, avg_terms, avg_terms)
            observed_data = observed_data.transpose(1, 2)

            data_to_predict = data_to_predict.transpose(1, 2)
            data_to_predict = torch.nn.functional.avg_pool1d(
                data_to_predict, avg_terms, avg_terms)
            data_to_predict = data_to_predict.transpose(1, 2)

            tp_to_predict = tp_to_predict.transpose(1, 2)
            tp_to_predict = torch.nn.functional.avg_pool1d(
                tp_to_predict, avg_terms, avg_terms)
            tp_to_predict = tp_to_predict.transpose(1,
                                                    2).squeeze().unsqueeze(0)

            observed_tp = observed_tp.transpose(1, 2)
            observed_tp = torch.nn.functional.avg_pool1d(
                observed_tp, avg_terms, avg_terms)
            observed_tp = observed_tp.transpose(1, 2).squeeze().unsqueeze(0)

        self.observed_data = observed_data
        self.data_to_predict = data_to_predict
        self.tp_to_predict = tp_to_predict
        self.observed_tp = observed_tp

        # get available forecasts if possible
        self.avail_fcst = False if avail_fcst_feature is None else True
        if self.avail_fcst:
            self.available_forecasts = trajs[:, observe_steps:,
                                             avail_fcst_feature]

    def __len__(self):
        return len(self.observed_data)

    def __getitem__(self, index):
        if self.avail_fcst:
            return self.observed_data[index], self.data_to_predict[
                index], self.available_forecasts[
                    index], self.tp_to_predict, self.observed_tp
        else:
            return self.observed_data[index], self.data_to_predict[
                index], torch.tensor(
                    torch.nan), self.tp_to_predict, self.observed_tp


def collate_fn(data):
    observed_data, data_to_pred, available_forecasts, tp_to_predict, observed_tp = zip(
        *data)
    observed_data = torch.stack(observed_data)
    data_to_pred = torch.stack(data_to_pred)
    available_forecasts = torch.stack(available_forecasts)
    tp_to_predict = tp_to_predict[0]
    observed_tp = observed_tp[0]

    # filter nan in the available forecasts (unaligned resolution)
    not_nan_idx = torch.logical_not(torch.isnan(available_forecasts))
    available_forecasts = available_forecasts[not_nan_idx].reshape(
        observed_data.shape[0], -1, available_forecasts.shape[-1])
    if available_forecasts.numel() == 0:
        available_forecasts = None

    # observe_steps = observed_data.shape[1]
    data_dict = {
        "observed_data": observed_data,
        "data_to_predict": data_to_pred,
        "available_forecasts": available_forecasts,
        "observed_tp": observed_tp,
        "tp_to_predict": tp_to_predict,
        "observed_mask": None,
        "mask_predicted_data": None,
        "labels": None,
        "mode": "extrap"
    }
    return data_dict


# electricity load dataset
def mfred(double=False, window_width=24 * 12 * 2):
    df = pd.read_csv("../datasets/MFRED_wiztemp.csv",
                     parse_dates=True,
                     index_col=0).values
    trajs = np.lib.stride_tricks.sliding_window_view(df, window_width, axis=0)
    trajs = trajs.transpose(0, 2, 1)[::12]

    t = torch.arange(window_width)
    time = torch.diff(t).sum()
    sample_rate = window_width / time
    if double:
        t = t.double()
        trajs = torch.from_numpy(trajs).double()
    else:
        t = t.float()
        trajs = torch.from_numpy(trajs).float()
    features = {
        "hist_feature": [0],
        "fcst_feature": [0],
        # "avail_fcst_feature": None,
        "avail_fcst_feature": [1],
    }
    return trajs, t.unsqueeze(0), sample_rate, features


# wind power dataset
def nrel(double=False, window_width=24 * 12 * 2, transformed=False):
    df = pd.read_csv("../datasets/nrel_all.csv", parse_dates=True,
                     index_col=0).values
    trajs = np.lib.stride_tricks.sliding_window_view(df, window_width, axis=0)
    trajs = trajs.transpose(0, 2, 1)[::12]

    t = torch.arange(window_width)
    time = torch.diff(t).sum()
    sample_rate = window_width / time
    if double:
        t = t.double()
        trajs = torch.from_numpy(trajs).double()
    else:
        t = t.float()
        trajs = torch.from_numpy(trajs).float()
    features = {
        "hist_feature": [0],
        "fcst_feature": [0],
        # "avail_fcst_feature": None,
        "avail_fcst_feature": [1, 2],
    }
    return trajs, t.unsqueeze(0), sample_rate, features


# GEFCom dataset
def GEFCom(double=False, window_width=24 * 7 * 2, observe_steps = 24 * 7, zone = 'CT', transformed=False):

    df1 = pd.read_csv('../datasets/gefcom2017.csv', parse_dates=True)
    df1 = df1[[ 'ts', 'zone', 'demand', 'drybulb', 'dewpnt' ]]
    start = pd.Timestamp(2010, 1, 1)
    end = pd.Timestamp(2017, 1, 1)
    df1['ts'] = pd.to_datetime(df1['ts'])
    df1 = df1.loc[df1['ts'].apply(lambda x: start<= x < end )]

    df2 = pd.read_csv('../datasets/ISONE2017-2024.csv', parse_dates=True)
    df2 = df2[[ 'ts', 'zone', 'demand', 'drybulb', 'dewpnt' ]]
    df2['ts'] = pd.to_datetime(df2['ts'])

    df_ct1 = df1.groupby('zone').get_group(zone)
    df_ct2 = df2.groupby('zone').get_group(zone)
    df_ct = pd.concat([df_ct1, df_ct2])

    train_start_time = pd.Timestamp(2015, 1, 1)
    test_end_time = pd.Timestamp(2019, 1, 1)

    df_ct = df_ct.loc[df_ct['ts'].apply(lambda x: train_start_time <= x < test_end_time )]

    # 由露点温度和干球温度转换成摄氏温度后，计算相对湿度
    df_ct['RH'] = 100 * ( ((17.27 * ((df_ct['dewpnt'] - 32 )/1.8) ) / (
        ((df_ct['dewpnt'] - 32 )/1.8) + 237.3)).apply(math.exp) / (
        (17.27 * ((df_ct['drybulb'] - 32 )/1.8) ) / (
            ((df_ct['drybulb'] - 32 )/1.8) + 237.3)).apply(math.exp) )

    df_ct = df_ct[['demand', 'drybulb', 'RH']]
    df_ct = df_ct.values

    print(df_ct.shape)

    trajs = np.lib.stride_tricks.sliding_window_view(df_ct, window_width, axis=0)
    trajs = trajs.transpose(0, 2, 1)
    trajs = np.concatenate((trajs[:-8760, :, :], trajs[-8760 + int(window_width - observe_steps) - 1:, :, :][::int(window_width - observe_steps)]), axis = 0 )
    # trajs = np.concatenate(( trajs[:-8760, :, :][::], trajs[-8760 + int(window_width - observe_steps)-1:, :, :][::] ), axis = 0 )
    trajs = np.copy(trajs)

    print(trajs.shape)

    t = torch.arange(window_width)
    time = torch.diff(t).sum()
    sample_rate = window_width / time
    if double:
        t = t.double()
        trajs = torch.from_numpy(trajs).double()
    else:
        t = t.float()
        trajs = torch.from_numpy(trajs).float()
    features = {
        "hist_feature": [0],
        "fcst_feature": [0],
        # "avail_fcst_feature": None,
        "avail_fcst_feature": [1, 2],
    }
    return trajs, t.unsqueeze(0), sample_rate, features


def generate_data_set(name,
                      device,
                      double=False,
                      batch_size=128,
                      extrap=0,
                      trajectories_to_sample=100,
                      percent_missing_at_random=0.0,
                      normalize=True,
                      test_set_out_of_distribution=True,
                      noise_std=None,
                      t_nsamples=200,
                      observe_stride=1,
                      predict_stride=1,
                      avail_fcst_stride=12,
                      add_external_feature=False,
                      observe_steps=200,
                      seed=0,
                      avg_terms=1,
                      zone = 'CT',
                      **kwargs):
    setup_seed(seed)

    window_width=kwargs.get("window_width")
    
    if name == "nrel":
        trajectories, t, sample_rate, feature = nrel(
            double,
            transformed=kwargs.get("transformed"),
            window_width=kwargs.get("window_width"))

    elif name == "mfred":
        trajectories, t, sample_rate, feature = mfred(
            double, window_width=kwargs.get("window_width"))

    elif name == "GEFCom":
        trajectories, t, sample_rate, feature = GEFCom(
            double, window_width=kwargs.get("window_width"), observe_steps=observe_steps, zone = zone)

    else:
        raise ValueError("Unknown Dataset To Test")

    if not add_external_feature:
        feature["avail_fcst_feature"] = None

    if not extrap:
        bool_mask = torch.FloatTensor(
            *trajectories.shape).uniform_() < (1.0 - percent_missing_at_random)
        if double:
            float_mask = (bool_mask).double()
        else:
            float_mask = (bool_mask).float()
        trajectories = float_mask * trajectories

    if noise_std:
        trajectories += torch.randn(trajectories.shape) * noise_std


    if name != "GEFCom":
        train_split = int(0.8 * trajectories.shape[0])
        test_split = int(0.9 * trajectories.shape[0])
    else:
        # train_split = int( (trajectories.shape[0] - 8760/int(window_width - observe_steps))*0.9 )
        # test_split = int(trajectories.shape[0] - 8760/int(window_width - observe_steps))

        train_split = int(0.9 * ( (26300 - window_width) + 1))
        test_split = int( (26300 - window_width) + 1)

    if test_set_out_of_distribution:
        train_trajectories = trajectories[:train_split, :, :]
        val_trajectories = trajectories[train_split:test_split, :, :]
        test_trajectories = trajectories[test_split:, :, :]
        train_t = t
        val_t = t
        test_t = t
        print('train shape',  train_trajectories.shape)
        print('val shape',  val_trajectories.shape)
        print('test shape',  test_trajectories.shape)
       
    else:
        traj_index = torch.randperm(trajectories.shape[0])
        train_trajectories = trajectories[traj_index[:train_split], :, :]
        val_trajectories = trajectories[
            traj_index[train_split:test_split], :, :]
        test_trajectories = trajectories[traj_index[test_split:], :, :]
        train_t = t
        val_t = t
        test_t = t
        print('train shape',  train_trajectories.shape)
        print('val shape',  val_trajectories.shape)
        print('test shape',  test_trajectories.shape)

    if normalize:
        len_train, len_val, len_test = len(train_trajectories), len(
            val_trajectories), len(test_trajectories)
        dim = trajectories.shape[2]
        train_mean = torch.reshape(train_trajectories, (-1, dim)).cpu().numpy()
        train_mean = torch.from_numpy(np.nanmean(train_mean, axis=0))
        train_std = torch.reshape(train_trajectories, (-1, dim)).cpu().numpy()
        train_std = torch.from_numpy(np.nanstd(train_std, axis=0))
        train_trajectories = (torch.reshape(train_trajectories, (-1, dim)) -
                              train_mean) / train_std
        val_trajectories = (torch.reshape(val_trajectories,
                                          (-1, dim)) - train_mean) / train_std
        test_trajectories = (torch.reshape(test_trajectories,
                                           (-1, dim)) - train_mean) / train_std
        train_trajectories = train_trajectories.reshape((len_train, -1, dim))
        val_trajectories = val_trajectories.reshape((len_val, -1, dim))
        test_trajectories = test_trajectories.reshape((len_test, -1, dim))
    else:
        train_std = 1
        train_mean = 0

    rand_idx = torch.randperm(len(train_trajectories)).tolist()
    train_trajectories = train_trajectories[rand_idx]
    dltrain = DataLoader(TimeSeriesDataset(train_trajectories, train_t,
                                           observe_steps, avg_terms,
                                           **feature),
                         batch_size=batch_size,
                         shuffle=False,
                         collate_fn=collate_fn)
    dlval = DataLoader(TimeSeriesDataset(val_trajectories, val_t,
                                         observe_steps, avg_terms, **feature),
                       batch_size=batch_size,
                       shuffle=False,
                       collate_fn=collate_fn)
    dltest = DataLoader(TimeSeriesDataset(test_trajectories, test_t,
                                          observe_steps, avg_terms, **feature),
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_fn)

    b = next(iter(dltrain))
    if b["available_forecasts"] is not None:
        input_dim = (b["observed_data"].shape[-1],
                     b["available_forecasts"].shape[-1])
        avail_fcst_timesteps = b["available_forecasts"].shape[1]
        avail_fcst_dim = b["available_forecasts"].shape[2]
        print('avail_fcst_dim', avail_fcst_dim)
    else:
        input_dim = (b["observed_data"].shape[-1], None)
        avail_fcst_timesteps = None

    output_dim = b["data_to_predict"].shape[-1]
    input_timesteps = (b["observed_data"].shape[1], avail_fcst_timesteps)
    output_timesteps = b["data_to_predict"].shape[1]

    return (input_dim, output_dim, sample_rate, t, dltrain, dlval, dltest,
            input_timesteps, output_timesteps, train_mean, train_std, feature)


def generate_tree_data_set(
    name,
    device,
    double=False,
    batch_size=128,
    extrap=0,
    trajectories_to_sample=100,
    percent_missing_at_random=0.0,
    normalize=True,
    test_set_out_of_distribution=True,
    noise_std=None,
    t_nsamples=200,
    observe_stride=1,
    predict_stride=1,
    avail_fcst_stride=12,
    add_external_feature=False,
    observe_steps=200,
    seed=0,
    avg_terms=1,
    zone = 'CT', 
    **kwargs,
):
    setup_seed(seed)

    window_width=kwargs.get("window_width")

    if name == "nrel":
        trajectories, t, sample_rate, feature = nrel(
            double,
            transformed=kwargs.get("transformed"),
            window_width=kwargs.get("window_width"),
        )

    elif name == "mfred":
        trajectories, t, sample_rate, feature = mfred(
            double, window_width=kwargs.get("window_width"))

    elif name == "GEFCom":
        trajectories, t, sample_rate, feature = GEFCom(
            double, window_width=kwargs.get("window_width"), observe_steps=observe_steps, zone = zone)
        
    else:
        raise ValueError("Unknown Dataset To Test")

    if not add_external_feature:
        feature["avail_fcst_feature"] = None

    if not extrap:
        bool_mask = torch.FloatTensor(
            *trajectories.shape).uniform_() < (1.0 - percent_missing_at_random)
        if double:
            float_mask = (bool_mask).double()
        else:
            float_mask = (bool_mask).float()
        trajectories = float_mask * trajectories

    if noise_std:
        trajectories += torch.randn(trajectories.shape) * noise_std

    if name != "GEFCom":
        train_split = int(0.8 * trajectories.shape[0])
        test_split = int(0.9 * trajectories.shape[0])
    else:
        # train_split = int( (trajectories.shape[0] - 8760/int(window_width - observe_steps))*0.9 )
        # test_split = int(trajectories.shape[0] - 8760/int(window_width - observe_steps))

        train_split = int(0.9 * ( (26300 - window_width) + 1))
        test_split = int( (26300 - window_width) + 1)

    if test_set_out_of_distribution:
        train_trajectories = trajectories[:train_split, :, :]
        val_trajectories = trajectories[train_split:test_split, :, :]
        test_trajectories = trajectories[test_split:, :, :]
        if name.__contains__("time"):
            train_t = t[:train_split]
            val_t = t[train_split:test_split]
            test_t = t[test_split:]
        else:
            train_t = t
            val_t = t
            test_t = t

        print('train size', train_trajectories.shape)
        print('val size', val_trajectories.shape)
        print('test size', test_trajectories.shape)


    else:
        traj_index = torch.randperm(trajectories.shape[0])
        train_trajectories = trajectories[traj_index[:train_split], :, :]
        val_trajectories = trajectories[
            traj_index[train_split:test_split], :, :]
        test_trajectories = trajectories[traj_index[test_split:], :, :]
        if name.__contains__("time"):
            train_t = t[traj_index[:train_split]]
            val_t = t[traj_index[train_split:test_split]]
            test_t = t[traj_index[test_split:]]
        else:
            train_t = t
            val_t = t
            test_t = t

        print('train size', train_trajectories.shape)
        print('val size', val_trajectories.shape)
        print('test size', test_trajectories.shape)
        
    if normalize:
        len_train, len_val, len_test = (
            len(train_trajectories),
            len(val_trajectories),
            len(test_trajectories),
        )
        dim = trajectories.shape[2]
        train_mean = torch.reshape(train_trajectories, (-1, dim)).cpu().numpy()
        train_mean = torch.from_numpy(np.nanmean(train_mean, axis=0))
        train_std = torch.reshape(train_trajectories, (-1, dim)).cpu().numpy()
        train_std = torch.from_numpy(np.nanstd(train_std, axis=0))
        train_trajectories = (torch.reshape(train_trajectories, (-1, dim)) -
                              train_mean) / train_std
        val_trajectories = (torch.reshape(val_trajectories,
                                          (-1, dim)) - train_mean) / train_std
        test_trajectories = (torch.reshape(test_trajectories,
                                           (-1, dim)) - train_mean) / train_std
        train_trajectories = train_trajectories.reshape((len_train, -1, dim))
        val_trajectories = val_trajectories.reshape((len_val, -1, dim))
        test_trajectories = test_trajectories.reshape((len_test, -1, dim))
    else:
        train_std = 1
        train_mean = 0

    rand_idx = torch.randperm(len(train_trajectories)).tolist()
    train_trajectories = train_trajectories[rand_idx]

    dltrain = DataLoader(
        TimeSeriesDataset(train_trajectories, train_t, observe_steps,
                          avg_terms, **feature),
        batch_size=len(train_trajectories),
        shuffle=False,
        collate_fn=collate_fn,
    )
    dlval = DataLoader(
        TimeSeriesDataset(val_trajectories, val_t, observe_steps, avg_terms,
                          **feature),
        batch_size=len(train_trajectories),
        shuffle=False,
        collate_fn=collate_fn,
    )
    dltest = DataLoader(
        TimeSeriesDataset(test_trajectories, test_t, observe_steps, avg_terms,
                          **feature),
        batch_size=len(train_trajectories),
        shuffle=False,
        collate_fn=collate_fn,
    )

    train_data = next(iter(dltrain))
    val_data = next(iter(dlval))
    test_data = next(iter(dltest))

    x_train = torch.concat(
        [
            train_data["observed_data"].flatten(1),
            train_data["available_forecasts"].flatten(1),
        ],
        dim=-1,
    )
    y_train = train_data["data_to_predict"].flatten(1)
    x_val = torch.concat(
        [
            val_data["observed_data"].flatten(1),
            val_data["available_forecasts"].flatten(1),
        ],
        dim=-1,
    )
    y_val = val_data["data_to_predict"].flatten(1)
    x_test = torch.concat(
        [
            test_data["observed_data"].flatten(1),
            test_data["available_forecasts"].flatten(1),
        ],
        dim=-1,
    )
    y_test = test_data["data_to_predict"].flatten(1)
    dltrain = (x_train.cpu().numpy(), y_train.cpu().numpy())
    dlval = (x_val.cpu().numpy(), y_val.cpu().numpy())
    dltest = (x_test.cpu().numpy(), y_test.cpu().numpy())

    input_dim = x_train.shape[-1]
    output_dim = y_train.shape[-1]

    return (
        input_dim,
        output_dim,
        sample_rate,
        t,
        dltrain,
        dlval,
        dltest,
        None,
        None,
        train_mean,
        train_std,
        feature,
    )
