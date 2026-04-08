from setuptools import find_packages, setup

def get_requirements(file_path):
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name="first_ml_project",
    version="0.0.1",
    author="Mohnish",
    author_email="mohnishshandilya.3000@gmail.com"
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)