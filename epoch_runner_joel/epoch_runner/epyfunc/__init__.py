import subprocess
import time
from typing import Literal, Dict, Tuple

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from quantiphy import Quantity


class SlurmJob:
    """Create and poll a SLURM Array job until completion

    Arguments
    ---------
    verbose:
        Whether to print out the running jobs

    Examples
    --------
    >>> job = epyfunc.SlurmJob(args.verbose)
    >>> job.enqueue_array_job(...)
    >>> job.poll_jobs(interval=2)
    >>> job_results = job.get_job_results()
    ({'10138561_2': 'COMPLETED'}), ({'10138561_1': 'FAILED'})
    """

    def __init__(self, verbose: bool = False) -> None:
        self.job_id = None
        self.verbose = verbose

    def _format_slurm_jobs(
        self, process: subprocess.CompletedProcess
    ) -> Dict[str, Literal["PENDING", "RUNNING", "COMPLETING"]]:
        """Format the job process into a Dict

        Arguments
        ---------
        process:
            the result from subprocess

        Returns
        -------
        Dict:
            The job id and status of the simulation (See https://curc.readthedocs.io/en/latest/running-jobs/squeue-status-codes.html)
        """
        job_statuses = {}
        for line in process.stdout.strip().split("\n"):
            if line:
                job_id, status = line.split()
                # Only add the actual jobs and not the job_id.+ or job_id.0 jobs
                if "." not in job_id:
                    job_statuses[job_id] = status

        if self.verbose:
            print(job_statuses)

        return job_statuses

    def enqueue_array_job(
        self,
        epoch_path: str,
        epoch_version: Literal["epoch1d", "epoch2d", "epoch3d"],
        campaign_path: str,
        file_path: str,
        template_path: str,
        n_runs: str,
        job_name: str = "jobscript",
    ) -> str:
        """Run a simulation using the specified path

        Arguments
        ---------
        epoch_path:
            The path to the main epoch directory
        epoch_version:
            The version of epoch to run
        campaign_path:
            The path of the campaign
        file_path:
            The path to the deck files
        template_path:
            The path to the template jobscript file
        n_runs:
            The number of simulations that need to run
        job_name:
            The name of the job to run

        Returns
        -------
        subprocess.CompletedProcess:
            The process object containing information about the completed process

        Examples
        --------
        >>> enqueue_simulations(
            epoch_path="/users/bmp535/scratch/epoch",
            epoch_version="epoch2d",
            deck_path="/users/bmp535/scratch/muons_test_e23",
            template_path="/users/bmp535/scratch/template.sh",
            job_name="muons_test_e23"
        )
        """
        with open(template_path) as f:
            s = f.read()

        s = s.format(
            job_name=job_name,
            campaign_path=campaign_path,
            array_range_max=n_runs,
            epoch_dir=epoch_path,
            epoch_version=epoch_version,
            file_path=file_path,
        )

        with open(f"{campaign_path}/jobscript.sh", "w") as f:
            f.write(s)

        process = subprocess.run(
            ["sbatch", f"{campaign_path}/jobscript.sh"],
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise RuntimeError(f"Error submitting job: {process.stderr}")

        # Extract job ID from the output
        self.job_id = process.stdout.split()[-1]

        if self.verbose:
            print(f"submitted array job with {self.job_id}")

        return self.job_id

    def get_running_jobs(
        self,
    ) -> Dict[str, Literal["PENDING", "RUNNING", "COMPLETING"]]:
        """Get runninng SLURM jobs

        Returns
        -------
        dict:
            The job id and status of the simulation
            (See https://curc.readthedocs.io/en/latest/running-jobs/squeue-status-codes.html)

        Examples
        --------
        >>> poll_simulation("123456")
        dict({"123456": "RUNNING"})
        """
        process = subprocess.run(
            ["squeue", "--job", self.job_id, "--Format=jobid,state", "--noheader"],
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise RuntimeError(f"Error getting running job status: {process.stderr}")

        return self._format_slurm_jobs(process=process)

    def get_job_results(
        self,
    ) -> Tuple[Dict[str, Literal["COMPLETED"]], Dict[str, Literal["FAILED"]]]:
        """Get all SLURM job results

        Returns
        -------
        dict, dict:
            Two sets of dictionaries containing completed and failed jobs respectively
        """
        process = subprocess.run(
            ["sacct", "--job", self.job_id, "--format=jobid,state", "--noheader"],
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise RuntimeError(f"Error getting completed job status: {process.stderr}")

        job_results = self._format_slurm_jobs(process=process)

        completed_jobs = {
            job_id: status
            for job_id, status in job_results.items()
            if status == "COMPLETED"
        }
        failed_jobs = {
            job_id: status
            for job_id, status in job_results.items()
            if status == "FAILED"
        }

        return completed_jobs, failed_jobs

    def poll_jobs(self, interval: int = 30) -> None:
        """Poll the SLURM jobs until they have finished running

        Arguments
        ---------
        interval:
            The interval at which to poll the running job

        Returns
        -------
        None:
            Runs until completion

        """
        while True:
            running_jobs = self.get_running_jobs()

            # Once the jobs are all completed, querying the jobs will simply return an empty dict
            if not running_jobs:
                return

            if self.verbose:
                print(f"Polling again in {interval} seconds...")

            time.sleep(interval)


def _get_frame_title(dataset: xr.Dataset, frame: int, display_sdf_name: bool) -> str:
    sdf_name = f"{frame:04d}.sdf" if display_sdf_name else ""
    time = dataset.isel(time=frame)["time"].values
    return f"$t = {Quantity(time, 's').render(prec=3)}$, {sdf_name}"


def generate_animation(
    dataset: xr.Dataset,
    target_attribute: str,
    folder_path: str = None,
    display: bool = False,
    display_sdf_name: bool = False,
    fps: int = 10,
    x_axis_coord: str = "X_Grid_mid",
    y_axis_coord: str = "Y_Grid_mid",
) -> HTML | None:
    """Generate an animation for the given target attribute

    Arguments
    ---------
        dataset:
            The dataset containing the simulation data
        target_attribute:
            The attribute to plot for each timestep
        folder_path:
            The path to save the generated animation (default: None)
        display:
            Whether to display the animation in the notebook (default: False)
        display_sdf_name:
            Display the sdf file name in the animation title
        fps:
            Frames per second for the animation (default: 10)
        x_axis_coord:
            Coordinate of the x-axis (default: "X_Grid_mid")
        y_axis_coord:
            Coordinate of the y-axis (default: "Y_Grid_mid")

    Examples
    --------
    >>> generateAnimation(dataset, "Derived_Number_Density_Electron")
    """
    fig, ax = plt.subplots()
    final_iteration = dataset.sizes.get("time")

    # Compute 1st and 99th percentiles to exclude extreme outliers
    global_min = np.percentile(dataset[target_attribute].values, 1)
    global_max = np.percentile(dataset[target_attribute].values, 99)

    norm = plt.Normalize(vmin=global_min, vmax=global_max)

    # Initialize the plot with the first timestep
    plot = dataset.isel(time=0)[target_attribute].plot(
        x=x_axis_coord, y=y_axis_coord, ax=ax, norm=norm, add_colorbar=False
    )
    title = _get_frame_title(dataset, 0, display_sdf_name)
    ax.set_title(title)
    cbar = plt.colorbar(plot, ax=ax)
    cbar.set_label(
        f'{dataset[target_attribute].attrs.get("long_name")} [${dataset[target_attribute].attrs.get("units")}$]'
    )

    def update(frame):
        ax.clear()
        plot = dataset.isel(time=frame)[target_attribute].plot(
            x=x_axis_coord, y=y_axis_coord, ax=ax, norm=norm, add_colorbar=False
        )
        title = _get_frame_title(dataset, frame, display_sdf_name)
        ax.set_title(title)
        cbar.update_normal(plot)

    ani = FuncAnimation(
        fig,
        update,
        frames=range(final_iteration),
        interval=1000 / fps,
        repeat=True,
    )

    # Save the animation
    if folder_path:
        try:
            ani.save(
                f"{folder_path}/{target_attribute.replace('/', '_')}.mp4",
                writer="ffmpeg",
                fps=fps,
            )
            print(
                f"Animation saved as MP4 at {folder_path}/{target_attribute.replace('/', '_')}.mp4"
            )
        except Exception as e:
            print(f"Failed to save as MP4 due to {e}. Falling back to GIF.")
            # Save as HTML
            ani.save(
                f"{folder_path}/{target_attribute.replace('/', '_')}.gif",
                writer="pillow",
                fps=fps,
            )
            print(
                f"Animation saved as GIF at {folder_path}/{target_attribute.replace('/', '_')}.mp4"
            )

    # Close the figure to avoid displaying the first frame as a separate plot
    plt.close(fig)

    if display:
        return HTML(ani.to_jshtml())
