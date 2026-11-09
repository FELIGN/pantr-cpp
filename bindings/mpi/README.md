# pantr-mpi

The compiled MPI glue behind `pantr.mpi`, the optional MPI-parallel layer of
[pantr](https://github.com/FELIGN/pantr). It is not meant to be installed on its own:

```bash
pip install "pantr[mpi]"
```

installs it together with `pantr` and `mpi4py`. The MPI library itself is not bundled; install
one of the family the wheel was built for (a system MPI, or the `mpich` / `openmpi` wheels).
`pantr.mpi` checks it before loading the extension.

pantr-mpi is released together with pantr, at the same version.
