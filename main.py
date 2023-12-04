from graphcast import graphcast
from graphcast import checkpoint
import numpy as np
import xarray
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Optional
import math
from IPython.display import HTML
import datetime
import webbrowser
from graphcast import data_utils
import dataclasses
import haiku as hk
import jax
import functools
from graphcast import rollout
from graphcast import casting
from graphcast import normalization
from graphcast import autoregressive

GRAPHCAST_SMALL_DATA_SOURCE = "data_files/modeltypes/params_GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
GRAPHCAST_SMALL_SAMPLE_DATA = "data_files/valid_dataset_for_graphcast_small/dataset_source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc"
# GRAPHCAST_SMALL_SAMPLE_DATA = 'data_files/valid_dataset_for_graphcast_small/dataset_source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc'
DIFF_STANDARD_DEVIATION_BY_LEVEL = "data_files/stats/stats_diffs_stddev_by_level.nc"
MEAN_BY_LEVEL = "data_files/stats/stats_mean_by_level.nc"
STANDARD_DEVIATION_BY_LEVEL = "data_files/stats/stats_stddev_by_level.nc"

SOURCE_1 = "random"
SOURCE_2 = "checkpoint"
SOURCE_TO_USE = SOURCE_2

# CONSTANTS FOR GRAPHCAST MODEL CONFIG IS SOURCE = RANDOM
RANDOM_MESH_SIZE = 5
RANDOM_GNN_MSG_STEPS = 10
RANDOM_LATENT_SIZE = 32
RANDOM_LEVELS = 13

# CREATING MODEL CONFIG AND LOADING IT
if SOURCE_TO_USE == SOURCE_1:
    # IF SOURCE IS RANDOM THEN USE THE USER INPUT DATA (NOT CLEAR HOW THIS WORKS.)
    params = None
    state = {}
    model_config = graphcast.ModelConfig(
        resolution=0,
        mesh_size=RANDOM_MESH_SIZE,
        latent_size=RANDOM_LATENT_SIZE,
        gnn_msg_steps=RANDOM_GNN_MSG_STEPS,
        hidden_layers=1,
        radius_query_fraction_edge_length=0.6,
    )
    task_config = graphcast.TaskConfig(
        input_variables=graphcast.TASK.input_variables,
        target_variables=graphcast.TASK.target_variables,
        forcing_variables=graphcast.TASK.forcing_variables,
        pressure_levels=graphcast.PRESSURE_LEVELS[RANDOM_LEVELS],
        input_duration=graphcast.TASK.input_duration,
    )
else:
    # IF SOURCE IS CHECKPOINT THEN LOAD THE GRAPHCAST_SMALL MODEL.
    # SOURCE SHOULD BE CHECKPOINT
    assert SOURCE_TO_USE == SOURCE_2
    # OPEN GRAPHCAST SMALL FILE PRESENT IN THE LOCAL DIRECTORY

    # NOT SURE IF WE NEED THIS. WE CAN DIRECTLY LOAD THE FILE USING checkpoint.load
    with np.load(GRAPHCAST_SMALL_DATA_SOURCE) as data:
        print("data")
        print(data)
    ckpt = checkpoint.load(GRAPHCAST_SMALL_DATA_SOURCE, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    print("Model description:\n", ckpt.description, "\n")
    print("Model license:\n", ckpt.license, "\n")

print("checking resolution early")
print(model_config.resolution)
print(model_config)
print(task_config)

# VALID DATA FILE IS PRESENT UNDER data_files/valid_dataset_for_graphcast_small THEREFORE WE DO NOT NEED THIS.
#
# # LOAD THE EXAMPLE DATA
# def data_valid_for_model(
#     file_name: str, model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig):
#     file_parts = parse_file_parts(file_name.removesuffix(".nc"))
#     return (
#         model_config.resolution in (0, float(file_parts["res"])) and
#         len(task_config.pressure_levels) == int(file_parts["levels"]) and
#         (
#             ("total_precipitation_6hr" in task_config.input_variables and
#             file_parts["source"] in ("era5", "fake")) or
#             ("total_precipitation_6hr" not in task_config.input_variables and
#             file_parts["source"] in ("hres", "fake"))
#         )
# )

example_batch = xarray.load_dataset(GRAPHCAST_SMALL_SAMPLE_DATA).compute()
# NEDD TO FINDOUT WHY WE ARE CHECKING FOR TIME TO BE >= 3
assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

print("\nVariables\n")
print(example_batch.data_vars.keys())
print("\nLevels\n")
print(example_batch.coords["level"].values)
print("\nSetps\n")
print(example_batch.dims["time"])
print("\nDims\n")
print(example_batch.dims)

# MAKING VALUES AS CONSTANTS FOR PLOTTING FOR NOW, LATER THESE VALUES SHOULD BE COME AS AN INPUT FROM USER BASED ON THE MODEL TYPE AND DATA SOURCE AND TYPE SELECTED
PLOT_EXAMPLE_VARIABLE = "2m_temperature"
PLOT_EXAMPLE_LEVEL = 0
PLOT_EXAMPLE_ROBUST = True
PLAOT_EXAMPLE_MAX_STEPS = 3

PLOT_SIZE = 7


# xpoints = np.array([1, 8])
# ypoints = np.array([3, 10])
# plt.plot(xpoints, ypoints)
# plt.show()


# SELECT A PARTICULAR DATA FROM THE FILE THAT CONTAINS TYPES OF DATAS
# FOR EXAMPLE - 2m_temperature, geopotential_at_surface, temperature, ETC.
def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> xarray.Dataset:
    data = data[variable]
    if "batch" in data.dims:
        data = data.isel(batch=0)
    if (
        max_steps is not None
        and "time" in data.sizes
        and max_steps < data.sizes["time"]
    ):
        data = data.isel(time=range(0, max_steps))
    if level is not None and "level" in data.coords:
        data = data.sel(level=level)
    return data


# NO IDEA WHAT IS GOING ON HERE
def scale(
    data: xarray.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    return (
        data,
        matplotlib.colors.Normalize(vmin, vmax),
        ("RdBu_r" if center is not None else "viridis"),
    )


# HELPER PLOTTING FUNCTION
def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4,
) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
    first_data = next(iter(data.values()))[0]
    max_steps = first_data.sizes.get("time", 1)
    assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

    cols = min(cols, len(data))
    rows = math.ceil(len(data) / cols)
    figure = plt.figure(figsize=(plot_size * 2 * cols, plot_size * rows))
    figure.suptitle(fig_title, fontsize=16)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()

    images = []
    for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
        ax = figure.add_subplot(rows, cols, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(
            plot_data.isel(time=0, missing_dims="ignore"),
            norm=norm,
            origin="lower",
            cmap=cmap,
        )
        plt.colorbar(
            mappable=im,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            aspect=16,
            shrink=0.75,
            cmap=cmap,
            extend=("both" if robust else "neither"),
        )
        images.append(im)

    def update(frame):
        if "time" in first_data.dims:
            td = datetime.timedelta(
                microseconds=first_data["time"][frame].item() / 1000
            )
            figure.suptitle(f"{fig_title}, {td}", fontsize=16)
        else:
            figure.suptitle(fig_title, fontsize=16)
        for im, (plot_data, norm, cmap) in zip(images, data.values()):
            im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

    print("\n\nCHECKING ANIMATION DATA\n\n")
    print(figure)
    print(figure.number)
    print(max_steps)
    ani = animation.FuncAnimation(
        fig=figure, func=update, frames=max_steps, interval=250
    )
    animation_file = "animation.html"
    ani.save(animation_file, writer="html")
    webbrowser.open(animation_file)
    # plt.close(figure.number)
    # plt.show(figure.number)
    # return HTML(ani.to_jshtml())


data = {
    " ": scale(
        select(
            example_batch,
            PLOT_EXAMPLE_VARIABLE,
            PLOT_EXAMPLE_LEVEL,
            PLAOT_EXAMPLE_MAX_STEPS,
        ),
        robust=PLOT_EXAMPLE_ROBUST,
    ),
}
fig_title = PLOT_EXAMPLE_VARIABLE
print("\nChecking plot example variable in batch\n")
print(example_batch[PLOT_EXAMPLE_VARIABLE])
# if "level" in example_batch[PLOT_EXAMPLE_VARIABLE].coords:
#   fig_title += f" at {PLOT_EXAMPLE_LEVEL} hPa"

print("\nPrint the Scale and Selected data : \n")
print(data)

# PLOT EXAMPLE DATA LOADED FROM THE GOOGLE CLOUD PUBLIC BUCKET
plot_data(data, fig_title, PLOT_SIZE, PLOT_EXAMPLE_ROBUST)

TRAIN_SETPS = example_batch.sizes["time"] - 2  # I.E. TRAINING STEP IS 1
EVAL_SETPS = example_batch.sizes["time"] - 2  # I.E. TRAINING STEP IS 1

# TRAING THE MODEL

# FINDING THE TRAINING INPUTS
(
    train_inputs,
    train_targets,
    train_forcings,
) = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", f"{TRAIN_SETPS*6}h"),
    **dataclasses.asdict(task_config),
)

print("\n\n")
print("TRAINING DATA")
print("\n\n")
print(train_inputs)
print("\n\n")
print(train_targets)
print("\n\n")
print(train_forcings)
print("\n\n")

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", f"{TRAIN_SETPS*6}h"),
    **dataclasses.asdict(task_config),
)
print("\n\n")
print(eval_inputs)
print("\n\n")
print(eval_targets)
print("\n\n")
print(eval_forcings)
print("\n\n")

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)


diffs_stddev_by_level = xarray.load_dataset(DIFF_STANDARD_DEVIATION_BY_LEVEL).compute()
mean_by_level = xarray.load_dataset(MEAN_BY_LEVEL).compute()
stddev_by_level = xarray.load_dataset(STANDARD_DEVIATION_BY_LEVEL).compute()

print("\n\n")
print(diffs_stddev_by_level)
print("\n\n")
print(mean_by_level)
print("\n\n")
print(stddev_by_level)
print("\n\n")


def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig
):
    """Constructs and wraps the GraphCast Predictor."""
    # Deeper one-step predictor.
    predictor = graphcast.GraphCast(model_config, task_config)

    # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
    # from/to float32 to/from BFloat16.
    predictor = casting.Bfloat16Cast(predictor)

    # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
    # BFloat16 happens after applying normalization to the inputs/targets.
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )

    # Wraps everything so the one-step model can produce trajectories.
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(
        lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics),
    )


def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, jax.random.PRNGKey(0), model_config, task_config, i, t, f
        )
        return loss, (diagnostics, next_state)

    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(_aux, has_aux=True)(
        params, state, inputs, targets, forcings
    )
    return loss, diagnostics, next_state, grads


# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
    return functools.partial(fn, model_config=model_config, task_config=task_config)


# Always pass params and state, so the usage below are simpler
def with_params(fn):
    return functools.partial(fn, params=params, state=state)


# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
    return lambda **kw: fn(**kw)[0]


init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
    params, state = init_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=train_inputs,
        targets_template=train_targets,
        forcings=train_forcings,
    )

loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))


# RUNNING THE MODAL
print("model_config.resolution")
print(model_config.resolution)
print(eval_inputs.sizes["lon"])
print(360.0 / eval_inputs.sizes["lon"])
assert model_config.resolution in (0, 360.0 / eval_inputs.sizes["lon"]), (
    "Model resolution doesn't match the data resolution. You likely want to "
    "re-filter the dataset list, and download the correct data."
)

print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings,
)
predictions

# PLOT THE PREDICTIONS

PLOT_SIZE = 5
PLAOT_EXAMPLE_MAX_STEPS = min(predictions.dims["time"], PLAOT_EXAMPLE_MAX_STEPS)

data = {
    "Targets": scale(
        select(
            eval_targets,
            PLOT_EXAMPLE_VARIABLE,
            PLOT_EXAMPLE_LEVEL,
            PLAOT_EXAMPLE_MAX_STEPS,
        ),
        robust=PLOT_EXAMPLE_ROBUST,
    ),
    "Predictions": scale(
        select(
            predictions,
            PLOT_EXAMPLE_VARIABLE,
            PLOT_EXAMPLE_LEVEL,
            PLAOT_EXAMPLE_MAX_STEPS,
        ),
        robust=PLOT_EXAMPLE_ROBUST,
    ),
    "Diff": scale(
        (
            select(
                eval_targets,
                PLOT_EXAMPLE_VARIABLE,
                PLOT_EXAMPLE_LEVEL,
                PLAOT_EXAMPLE_MAX_STEPS,
            )
            - select(
                predictions,
                PLOT_EXAMPLE_VARIABLE,
                PLOT_EXAMPLE_LEVEL,
                PLAOT_EXAMPLE_MAX_STEPS,
            )
        ),
        robust=PLOT_EXAMPLE_ROBUST,
        center=0,
    ),
}


# Print the extracted data
fig_title = PLOT_EXAMPLE_VARIABLE
if "level" in predictions[PLOT_EXAMPLE_VARIABLE].coords:
    fig_title += f" at {PLOT_EXAMPLE_LEVEL} hPa"

plot_data(data, fig_title, PLOT_SIZE, PLOT_EXAMPLE_ROBUST)


from flask import Flask, jsonify, request
import numpy as np
import xarray as xr
from datetime import timedelta
app = Flask(__name__)

@app.route('/predicted_temperature', methods=['GET'])
def get_predicted_temperature():
    payload = request.get_json()

    desired_lat = payload.get("latitude")
    desired_lon = payload.get("longitude")
    desired_time = "06:00:00"


    # Select the temperature at the desired location and time
    before_desired_temperature = data["Targets"][0].sel(
        lat=desired_lat, lon=desired_lon, time=desired_time, method="nearest"
    )


    # Select the temperature at the desired location and time
    after_desired_temperature = data["Predictions"][0].sel(
        lat=desired_lat, lon=desired_lon, time=desired_time, method="nearest"
    )

    # Print the result
    print(
        f"Predicted temperature at {desired_lat}° latitude, {desired_lon}° longitude, and {desired_time}: {after_desired_temperature.values}"
    )

    # Create a response in JSON format
    celcius_temperature = round(after_desired_temperature.to_dict()["data"] - 273.15, 2)
    # Create a response in JSON format
    response = {
        'Expected temperature is ': str(celcius_temperature) + " Celcius",
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=8010)