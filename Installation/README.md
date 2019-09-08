# Installation

## Arch Linux


### Install yaourt

```console
$ sudo pacman -S --needed base-devel git wget yajl
$ git clone https://aur.archlinux.org/package-query.git
$ cd package-query/
$ makepkg -si
$ cd ..
$ git clone https://aur.archlinux.org/yaourt.git
$ cd yaourt/
$ makepkg -si
```

### Install SymPy with Version < 1.2
```console
$ wget https://archive.archlinux.org/packages/p/python-sympy/python-sympy-1.1.1-1-any.pkg.tar.xz
$ sudo mv python-sympy-1.1.1-1-any.pkg.tar.xz /var/cache/pacman/pkg/
$ sudo pacman -U /var/cache/pacman/pkg/python-sympy-1.1.1-1-any.pkg.tar.xz
$ sudo cp -rf /usr/lib/python3.6/site-packages/sympy-1.1.1-py3.6.egg-info /usr/lib/python3.7/
$ sudo cp -rf /usr/lib/python3.6/site-packages/sympy /usr/lib/python3.7/
```
Also be sure to add SymPy to the `IgnorePkg` list in `/etc/pacman.conf`.
```
IgnorePkg  =  python-sympy    
```

### Install FEniCS Packages
Install Dolfin from AUR via:
```console
$ yaourt -S dolfin
```
Check the final ouput of the installation and identify the file that should be sourced:
```console
source /usr/share/dolfin/dolfin.conf
```
This command should then be included in the `.bashrc` file to make the environment variable changes persistent.

Finally, install the Python packages:
```console
$ yaourt -S python-dolfin
$ yaourt -S python-mshr    
```


### Install TensorFlow
```console
sudo pacman -S tensorflow python-tensorflow tensorboard
```

Also, if the PyPng package is not already installed, it can be installed via:
```console
$ sudo pip install pypng
```



## Red Hat Enterprise Linux

Add external CentOS repository with the `container-selinux` package and install `docker-ce` (the freely available community edition):
```
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum makecache fast
sudo yum install -y http://mirror.centos.org/centos/7/extras/x86_64/Packages/container-selinux-2.95-2.el7_6.noarch.rpm
sudo yum install -y docker-ce
```

Check whether `docker` service is running and manually start if needed:
```
sudo service docker status
sudo systemctl start docker
```


### Executing the code using Docker

Change to the `ConvPDE` directory and run the `fenicsproject` container:
```
cd ~/git/ConvPDE/
sudo docker run --env HTTP_PROXY="[proxy]" --env HTTPS_PROXY="[proxy]" -ti -v $(pwd):/home/fenics/shared:z quay.io/fenicsproject/stable 
```
where `[proxy]` corresponds to the environment variable `HTTP_PROXY` (which can be displayed via `echo $HTTP_PROXY`).


Install `pypng` package for edge detection in mesh generation procedure:
```
pip3 install --user pypng   # NOTE: This must be run every time...
```

Run the setup script to execute the FEniCS portion of the code:
```
cd Poisson_Varying_Domain/Setup/
sudo chmod +x RUN_FEniCS.sh
sudo ./RUN_FEniCS.sh
```

Leave container, optionally enter a TensorFlow virtual environment (e.g. `~/Documents/virtual_envs/tf`), and execute the remaining setup script:
```
exit
source ~/Documents/virtual_envs/tf/bin/activate
sudo chmod +x RUN_TF.sh
sudo ./RUN_TF.sh
```

Update the permissions on the data generated, leave `Setup/` subdirectory, and run training script:
```
sudo chmod -R 777 ./
cd ..
chmod +x Train_Model.sh
./Train_Model.sh 1
```



### Random reference links
* [FEniCS Containers](https://buildmedia.readthedocs.org/media/pdf/fenics-containers/latest/fenics-containers.pdf)
* [FEniCS Installation](https://fenics.readthedocs.io/en/latest/installation.html#from-source)
* [Docker CE for RHEL 1](https://nickjanetakis.com/blog/docker-tip-39-installing-docker-ce-on-redhat-rhel-7x)
* [Docker CE for RHEL 2](https://stackoverflow.com/questions/45272827/docker-ce-on-rhel-requires-container-selinux-2-9/45274492#45274492)
* [CentOS Package List](http://mirror.centos.org/centos/7/extras/x86_64/Packages/)
* [Docker Proxy Settings 1](https://docs.docker.com/network/proxy/)
* [Docker Proxy Settings 2](https://groups.google.com/forum/#!topic/coreos-user/T7Lz8IT5NT4)
* [Docker Proxy Settings 3](https://forums.docker.com/t/issue-with-installing-pip-packages-inside-a-docker-container-with-ubuntu/35107)
* [Docker Daemon](https://forums.docker.com/t/cannot-connect-to-the-docker-daemon-is-the-docker-daemon-running-on-this-host/8925/3)
* [Docker External Files](https://stackoverflow.com/questions/30652299/having-docker-access-external-files)
