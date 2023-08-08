from pysteps import verification
import matplotlib.pyplot as plt
import numpy as np

def FSS_plot(forecasts, observed_precip, methods, start_index):

    precip_forecast, persistence_forecast , linda_forecast, steps_forecast = forecasts

    fss = verification.get_method("FSS")

    # Compute fractions skill score (FSS) for all lead times for different scales using a 1 mm/h detection threshold.
    scales = [
        2,
        # 4,
        # 8,
        # 16,
        # 32,
        # 64,
    ]  # In grid points.

    scales_in_km = np.array(scales) * 10

    # Set the threshold
    thr = 1.0  # in mm/h

    # Now plot it
    plt.figure(figsize=(10,10))
    # Calculate the FSS for every lead time and all predefined scales.
    for method in methods:
        score = []
        if method == 'Naive Persistence':
            prediction_forecast = persistence_forecast
        elif method == 'Lagrangian Persistence':
            prediction_forecast = precip_forecast
        elif method == 'LINDA':
            prediction_forecast = linda_forecast
        elif method == 'steps':
            prediction_forecast = steps_forecast
            


        for i in range(len(observed_precip)):
            score_ = []
            for scale in scales:
                score_.append(
                    fss(prediction_forecast[i, :, :], observed_precip[i, :, :], thr, scale)
                )
            score.append(score_)
        x = np.arange(1, len(observed_precip) + 1)
        plt.plot(x*30, score, lw=2.0, label=method)

    plt.xlabel("Lead time [min]")
    plt.ylabel("FSS ( > 1.0 mm/h ) ")
    plt.title("Fractions Skill Score")
    plt.legend(
        # scales_in_km,
        # title="Scale [km]",
        # loc="center left",
        # bbox_to_anchor=(1.01, 0.5),
        # bbox_transform=plt.gca().transAxes,
    )
    plt.autoscale(axis="x", tight=True)
    plt.tight_layout()
    plt.savefig('FSS_comparison_'+str(start_index)+'.png')