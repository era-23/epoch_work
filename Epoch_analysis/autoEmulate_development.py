# General imports for the notebook
import warnings

from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")
from autoemulate.simulations.projectile import Projectile
from autoemulate import AutoEmulate

projectile = Projectile(log_level="error")
n_samples = 500
x = projectile.sample_inputs(n_samples).float()
y, _ = projectile.forward_batch(x)
y = y.float()

print(x.shape)
print(y.shape)

print(AutoEmulate.list_emulators())

# Run AutoEmulate with default settings
ae = AutoEmulate(x, y, only_probabilistic = True, log_level="progress_bar")

print(ae.summarise())

best = ae.best_result()
print("Model with id: ", best.id, " performed best: ", best.model_name)

ae.plot(best, fname="best_model_plot.png")

ae.plot(best, input_ranges={0: (0, 4), 1: (200, 500)}, output_ranges={0: (0, 10)})

print(best.model.predict(x[:10]))