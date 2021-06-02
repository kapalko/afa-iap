from setuptools import setup

setup(name='airforce_arcade',
      version='0.0.1',
      python_requires=">=3",
      install_requires=[
      "gym", 
      "mlagents_envs == 0.22.0",
      "opencv-python"]
)