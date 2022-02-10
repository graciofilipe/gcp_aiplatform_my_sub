from setuptools import find_packages, setup

REQUIRED_PACKAGES = ["numpy", "pandas", "scikit-learn",
                     "tensorboard==2.0.2", "tensorflow==2.5.3",
                     "tensorflow-estimator==2.0.1", "gcsfs"]

setup(name="trainer_mirrored",
      version="0.1",
      install_requires=REQUIRED_PACKAGES,
      packages=find_packages(),
      include_package_data=True,
      description='my first attempt')
