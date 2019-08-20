from setuptools import setup
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

with open("VERSION.txt", "r") as f:
    VERSION = f.read().strip().strip('"')

setup(
    name="gym-forest",
    version=VERSION,
    author="Rigetti Computing",
    author_email="info@rigetti.com",
    description="Gym environments for quantum program synthesis",
    packages=["gym_forest",
              "gym_forest.envs"],
    include_package_data=True,
    # Dependent packages (distributions)
    install_requires=[str(req.req) for req in parse_requirements('requirements.txt', session='hack')],
)
