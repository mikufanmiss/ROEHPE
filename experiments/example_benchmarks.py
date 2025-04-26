import sys
sys.path.append("..")

import argparse
import logging
import pickle
from pathlib import Path
from time import strftime
import pandas as pd
import torch
import numpy as np

from sklearn.metrics import mean_squared_error
from benchmark.dataset import generate_data_set

from benchmark.model import GeneralNeuralLaplace
from benchmark.benchmarks import GeneralNeuralNetwork, GeneralPersistence
from benchmark.utils import train_and_test, setup_seed, init_weights

datasets = ["GEFCom"]

file_name = Path(__file__).stem


def experiment_with_all_baselines(
        dataset, double, batch_size, extrapolate, epochs, seed, run_times,
        learning_rate, weight_decay, trajectories_to_sample,
        time_points_to_sample, observe_stride, predict_stride, observe_steps,
        noise_std, normalize_dataset, encode_obs_time, latent_dim,
        hidden_units, avg_terms_list, patience, device,
        use_sphere_projection, ilt_algorithm, solete_energy, solete_resolution,
        transformed, window_width, add_external_feature, persistence, zone_list):
    # Compares against all baselines, 

    df_list_baseline_results = []

    for seed in range(seed, seed + run_times):
        setup_seed(seed)
        Path(f"./results/{dataset}").mkdir(parents=True, exist_ok=True)

        path = f"./results/{dataset}/{path_run_name}-{seed}.pkl"
        saved_dict = {}
        
        for zone in zone_list:

            (input_dim, output_dim, sample_rate, t, dltrain, dlval, dltest,
            input_timesteps, output_timesteps, train_mean, train_std,
            feature) = generate_data_set(
                dataset,
                device,
                double=double,
                batch_size=batch_size,
                trajectories_to_sample=trajectories_to_sample,
                extrap=extrapolate,
                normalize=normalize_dataset,
                noise_std=noise_std,
                t_nsamples=time_points_to_sample,
                observe_stride=observe_stride,
                predict_stride=predict_stride,
                observe_steps=observe_steps,
                seed=seed,
                solete_energy=solete_energy,
                solete_resolution=solete_resolution,
                transformed=transformed,
                add_external_feature=add_external_feature,
                window_width=window_width,
                avg_terms=1, 
                zone = zone)
            logger.info(f"input steps:\t {input_timesteps}")
            logger.info(f"output steps:\t {output_timesteps}")
            # desired_f = sample_rate / avg_terms
            # T = 2 * t.cpu().numpy().max()
            # s_terms = int(T * desired_f)
            # if s_terms % 2 == 0:
            #     s_terms += 1
            # logger.info(f"s_terms:\t {s_terms}")

            sub_saved_dict = {}
            # sub_saved_dict[f"avg_terms_{avg_terms}"] = {}
            # sub_saved_dict["dataset"] = dataset
            # sub_saved_dict["trajectories_to_sample"] = trajectories_to_sample
            # sub_saved_dict["extrapolate"] = extrapolate
            # sub_saved_dict["normalize_dataset"] = normalize_dataset
            sub_saved_dict["input_dim"] = input_dim
            sub_saved_dict["output_dim"] = output_dim
            sub_saved_dict["train_mean"] = train_mean
            sub_saved_dict["train_std"] = train_std

            # saved_dict[f"avg_terms_{avg_terms}"] = sub_saved_dict
            saved_dict[f"zone_{zone}"] = sub_saved_dict


            # Pre-save
            with open(path, "wb") as f:
                pickle.dump(saved_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            models = [
                (
                    "Neural Laplace",
                    GeneralNeuralLaplace(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        latent_dim=latent_dim,
                        hidden_units=hidden_units,
                        # s_recon_terms=s_terms,
                        s_recon_terms=33,
                        use_sphere_projection=use_sphere_projection,
                        include_s_recon_terms=True,
                        ilt_algorithm=ilt_algorithm,
                        encode_obs_time=encode_obs_time,
                        device=device,
                        encoder="rnn",
                        input_timesteps=input_timesteps,
                        output_timesteps=output_timesteps,
                        method="single").to(device),
                ),
                (
                    "LSTM",
                    GeneralNeuralNetwork(obs_dim=input_dim,
                                        out_dim=output_dim,
                                        out_timesteps=output_timesteps,
                                        in_timesteps=input_timesteps,
                                        nhidden=hidden_units,
                                        device=device,
                                        method="lstm").to(device),
                ),
                (
                    "MLP",
                    GeneralNeuralNetwork(obs_dim=input_dim,
                                        out_dim=output_dim,
                                        out_timesteps=output_timesteps,
                                        in_timesteps=input_timesteps,
                                        nhidden=hidden_units,
                                        device=device,
                                        method="mlp").to(device),
                ),
                (
                    "DLinear",
                    GeneralNeuralNetwork(obs_dim=input_dim,
                                        out_dim=output_dim,
                                        out_timesteps=output_timesteps,
                                        in_timesteps=input_timesteps,
                                        nhidden=hidden_units,
                                        device=device,
                                        method="dlinear").to(device),
                ),
                (
                    "NBEATSx",
                    GeneralNeuralNetwork(obs_dim=input_dim,
                                        out_dim=output_dim,
                                        out_timesteps=output_timesteps,
                                        in_timesteps=input_timesteps,
                                        nhidden=hidden_units,
                                        device=device,
                                        method="nbeats").to(device),
                ),
                (
                    "TFT",
                    GeneralNeuralNetwork(obs_dim=input_dim,
                                        out_dim=output_dim,
                                        out_timesteps=output_timesteps,
                                        in_timesteps=input_timesteps,
                                        nhidden=hidden_units,
                                        device=device,
                                        method="tft").to(device),
                ),
                (
                    "Persistence",
                    GeneralPersistence(out_timesteps=output_timesteps,
                                    out_feature=feature["fcst_feature"],
                                    device=device,
                                    method=persistence).to(device),
                ),
            ]

            for model_name, system in models:
                try:
                    logger.info(
                        f"Training & testing for : {model_name} \t | seed: {seed}"
                    )
                    if double:
                        system.double()
                    else:
                        system.float()
                    logger.info("num_params={}".format(
                        sum(p.numel() for p in system.model.parameters())))
                    if model_name != "Persistence":
                        if model_name != "TFT":
                            init_weights(system.model, seed)
                        optimizer = torch.optim.Adam(system.model.parameters(),
                                                    lr=learning_rate,
                                                    weight_decay=weight_decay)

                        scheduler = None
                        train_losses, val_losses, train_nfes, _ = train_and_test(
                            system,
                            dltrain,
                            dlval,
                            dltest,
                            optimizer,
                            device,
                            scheduler,
                            epochs=epochs,
                            patience=patience,
                        )
                    test_preds, test_trajs = system.predict(dltest)

                    logger.info("Prediction size of the model")
                    logger.info(test_preds.shape)
                    logger.info("The size of true load")
                    logger.info(test_preds.shape)
                    
                    label_std = train_std[0].cpu().numpy()
                    label_mean = train_mean[0].cpu().numpy()
                    test_pre = test_trajs.detach().cpu().numpy().flatten()
                    test_true = test_preds.detach().cpu().numpy().flatten()

                    test_rmse = np.sqrt(mean_squared_error(
                        test_true * label_std + label_mean,
                        test_pre * label_std + label_mean))
                    
                    test_mape = np.mean( np.abs( (test_true * label_std + label_mean) - 
                        (test_pre * label_std + label_mean) 
                        ) / (test_true * label_std + label_mean)) * 100
                    
                    daily_peak_rmse = np.sqrt(mean_squared_error(
                        np.max( (test_true).reshape(-1, 24), axis = 1 ) * label_std + label_mean,
                        np.max( (test_pre).reshape(-1, 24), axis = 1 ) * label_std + label_mean))
                    
                    daily_peak_mape = np.mean( np.abs( 
                        (np.max( (test_true).reshape(-1, 24), axis = 1 ) * label_std + label_mean) - 
                        (np.max( (test_pre).reshape(-1, 24), axis = 1 ) * label_std + label_mean)
                        ) / (np.max( (test_true).reshape(-1, 24), axis = 1 ) * label_std + label_mean)) * 100

                    logger.info(
                        f"zone seed: {zone}-{seed}")
                
                    logger.info(
                        f"Result: {model_name} - TEST RMSE: {test_rmse}")
                    
                    logger.info(
                        f"Result: {model_name} - TEST MAPE: {test_mape}")
                    
                    logger.info(
                        f"Result: {model_name} - Daily Peak TEST RMSE: {daily_peak_rmse}")
                    
                    logger.info(
                        f"Result: {model_name} - Daily Peak TEST MAPE: {daily_peak_mape}")
                        
                    df_list_baseline_results.append({
                        'method': model_name,
                        'zone': zone,
                        'test_rmse': test_rmse,
                        'test_mape': test_mape,
                        'daily_peak_rmse': daily_peak_rmse,
                        'daily_peak_mape': daily_peak_mape,
                        'seed': seed
                    })

                    sub_saved_dict[model_name] = {
                        "test rmse": test_rmse,
                        "test mape": test_mape,
                        'daily_peak_rmse': daily_peak_rmse,
                        'daily_peak_mape': daily_peak_mape,
                        "seed": seed,
                        "model_state_dict": system.model.state_dict(),
                        "train_losses": train_losses.detach().cpu().numpy(),
                        "val_losses": val_losses.detach().cpu().numpy(),
                        "train_nfes": train_nfes.detach().cpu().numpy(),
                        "test_preds": test_preds.detach().cpu().numpy(),
                        "test_trajs": test_trajs.detach().cpu().numpy(),
                    }
                    # Checkpoint
                    saved_dict[f"zone_{zone}"] = sub_saved_dict

                    with open(path, "wb") as f:
                        pickle.dump(saved_dict,
                                    f,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    pass
                    logger.error(e)
                    logger.error(f"Error for model: {model_name}")
                    raise e
            path = f"./results/{dataset}/{path_run_name}-{seed}.pkl"

            saved_dict[f"zone_{zone}"] = sub_saved_dict
            with open(path, "wb") as f:
                pickle.dump(saved_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Clear GPU memory after training
            torch.cuda.empty_cache()

    # Process results for experiment
    df_results = pd.DataFrame(df_list_baseline_results)
    test_rmse_df = df_results.groupby(['method', 'zone']).agg(["mean", "std"])["test_rmse"]
    test_mape_df = df_results.groupby(['method', 'zone']).agg(["mean", "std"])["test_mape"]
    daily_peak_test_rmse_df = df_results.groupby(['method', 'zone']).agg(["mean", "std"])["daily_peak_rmse"]
    daily_peak_test_mape_df = df_results.groupby(['method', 'zone']).agg(["mean", "std"])["daily_peak_mape"]
    logger.info("Test RMSE of experiment")
    logger.info(test_rmse_df)
    logger.info("Test MAPE of experiment")
    logger.info(test_mape_df)
    logger.info("Test Daily Peak RMSE of experiment")
    logger.info(daily_peak_test_rmse_df)
    logger.info("Test Daily Peak MAPE of experiment")
    logger.info(daily_peak_test_mape_df)
    return test_rmse_df, test_mape_df, daily_peak_test_rmse_df, daily_peak_test_mape_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Run all baselines for an experiment (including Neural Laplace)")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="sine",
        help=f"Available datasets: {datasets}",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--double", action="store_true")  # Default False
    parser.add_argument("--interpolate", action="store_false")  # Default True
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_times", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--trajectories_to_sample", type=int, default=1000)
    parser.add_argument("--time_points_to_sample", type=int, default=200)
    parser.add_argument("--observe_stride", type=int, default=1)
    parser.add_argument("--predict_stride", type=int, default=1)
    parser.add_argument("--observe_steps", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--normalize_dataset",
                        action="store_false")  # Default True
    parser.add_argument("--encode_obs_time",
                        action="store_true")  # Default False
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--hidden_units", type=int, default=42)

    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_sphere_projection",
                        action="store_false")  # Default True
    parser.add_argument("--ilt_algorithm", type=str, default="fourier")
    parser.add_argument("--solete_energy", type=str, default="solar")
    parser.add_argument("--solete_resolution", type=str, default="5min")
    parser.add_argument("--transformed", action="store_true")  # Default False
    parser.add_argument("--window_width", type=int,
                        default=24 * 12 * 2)  # Default False
    parser.add_argument("--add_external_feature", action="store_true")  # Default False

    parser.add_argument('--persistence', type=str, default="naive")
    parser.add_argument('--avg_terms_list', nargs='+', type=int, default=None)

    parser.add_argument('--zone_list', nargs='+', type=str, default=None)
    args = parser.parse_args()

    assert args.dataset in datasets
    device = torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    Path("./logs").mkdir(parents=True, exist_ok=True)
    path_run_name = "{}-{}".format(file_name, strftime("%Y%m%d-%H%M%S"))

    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(f"logs/{path_run_name}_log.txt"),
            logging.StreamHandler()
        ],
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger()

    logger.info(f"Using {device} device")

    test_rmse_df, test_mape_df, daily_peak_test_rmse_df, daily_peak_test_mape_df = experiment_with_all_baselines(
        dataset=args.dataset,
        double=args.double,
        batch_size=args.batch_size,
        extrapolate=args.interpolate,
        epochs=args.epochs,
        seed=args.seed,
        run_times=args.run_times,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        trajectories_to_sample=args.trajectories_to_sample,
        time_points_to_sample=args.time_points_to_sample,
        observe_stride=args.observe_stride,
        predict_stride=args.predict_stride,
        observe_steps=args.observe_steps,
        noise_std=args.noise_std,
        normalize_dataset=args.normalize_dataset,
        encode_obs_time=args.encode_obs_time,
        latent_dim=args.latent_dim,
        hidden_units=args.hidden_units,
        avg_terms_list=args.avg_terms_list,
        patience=args.patience,
        device=device,
        use_sphere_projection=args.use_sphere_projection,
        ilt_algorithm=args.ilt_algorithm,
        solete_energy=args.solete_energy,
        solete_resolution=args.solete_resolution,
        transformed=args.transformed,
        add_external_feature=args.add_external_feature,
        persistence=args.persistence,
        window_width=args.window_width, 
        zone_list = args.zone_list)


path = f"./results/{args.dataset}/test_std_dict-{path_run_name}-{args.seed}.pkl"
    
test_std_dict = {}
test_std_dict['test_rmse_df'] = test_rmse_df
test_std_dict['test_mape_df'] = test_mape_df
test_std_dict['daily_peak_test_rmse_df'] = daily_peak_test_rmse_df
test_std_dict['daily_peak_test_mape_df'] = daily_peak_test_mape_df

with open(path, "wb") as f:
    pickle.dump(test_std_dict, f, protocol=pickle.HIGHEST_PROTOCOL)