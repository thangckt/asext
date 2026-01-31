# {py:mod}`asext.io.readwrite`

```{py:module} asext.io.readwrite
```

```{autodoc2-docstring} asext.io.readwrite
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`read_extxyz <asext.io.readwrite.read_extxyz>`
  - ```{autodoc2-docstring} asext.io.readwrite.read_extxyz
    :summary:
    ```
* - {py:obj}`write_extxyz <asext.io.readwrite.write_extxyz>`
  - ```{autodoc2-docstring} asext.io.readwrite.write_extxyz
    :summary:
    ```
* - {py:obj}`read_lmpdump <asext.io.readwrite.read_lmpdump>`
  - ```{autodoc2-docstring} asext.io.readwrite.read_lmpdump
    :summary:
    ```
* - {py:obj}`write_lmpdata <asext.io.readwrite.write_lmpdata>`
  - ```{autodoc2-docstring} asext.io.readwrite.write_lmpdata
    :summary:
    ```
* - {py:obj}`extxyz2lmpdata <asext.io.readwrite.extxyz2lmpdata>`
  - ```{autodoc2-docstring} asext.io.readwrite.extxyz2lmpdata
    :summary:
    ```
* - {py:obj}`lmpdata2extxyz <asext.io.readwrite.lmpdata2extxyz>`
  - ```{autodoc2-docstring} asext.io.readwrite.lmpdata2extxyz
    :summary:
    ```
* - {py:obj}`lmpdump2extxyz <asext.io.readwrite.lmpdump2extxyz>`
  - ```{autodoc2-docstring} asext.io.readwrite.lmpdump2extxyz
    :summary:
    ```
* - {py:obj}`_get_symbols_by_types <asext.io.readwrite._get_symbols_by_types>`
  - ```{autodoc2-docstring} asext.io.readwrite._get_symbols_by_types
    :summary:
    ```
````

### API

````{py:function} read_extxyz(extxyz_file: str, index=':') -> list[ase.atoms.Atoms]
:canonical: asext.io.readwrite.read_extxyz

```{autodoc2-docstring} asext.io.readwrite.read_extxyz
```
````

````{py:function} write_extxyz(outfile: str, structs: list[ase.atoms.Atoms] | ase.atoms.Atoms) -> None
:canonical: asext.io.readwrite.write_extxyz

```{autodoc2-docstring} asext.io.readwrite.write_extxyz
```
````

````{py:function} read_lmpdump(lmpdump_file: str, index=-1, units='metal', **kwargs) -> list[ase.atoms.Atoms]
:canonical: asext.io.readwrite.read_lmpdump

```{autodoc2-docstring} asext.io.readwrite.read_lmpdump
```
````

````{py:function} write_lmpdata(file: str, atoms: ase.atoms.Atoms, *, specorder: list[str] | None = None, reduce_cell: bool = False, force_skew: bool = False, prismobj: ase.calculators.lammps.Prism | None = None, write_image_flags: bool = False, masses: bool = True, velocities: bool = False, units: str = 'metal', bonds: bool = True, atom_style: str = 'atomic') -> None
:canonical: asext.io.readwrite.write_lmpdata

```{autodoc2-docstring} asext.io.readwrite.write_lmpdata
```
````

````{py:function} extxyz2lmpdata(extxyz_file: str, lmpdata_file: str, masses: bool = True, units: str = 'metal', atom_style: str = 'atomic', **kwargs) -> tuple[list, list]
:canonical: asext.io.readwrite.extxyz2lmpdata

```{autodoc2-docstring} asext.io.readwrite.extxyz2lmpdata
```
````

````{py:function} lmpdata2extxyz(lmpdata_file: str, extxyz_file: str, original_cell_file: str | None = None)
:canonical: asext.io.readwrite.lmpdata2extxyz

```{autodoc2-docstring} asext.io.readwrite.lmpdata2extxyz
```
````

````{py:function} lmpdump2extxyz(lmpdump_file: str, extxyz_file: str, index: int | slice = -1, original_cell_file: str | None = None, stress_file: str | None = None, lammps_units: str = 'metal')
:canonical: asext.io.readwrite.lmpdump2extxyz

```{autodoc2-docstring} asext.io.readwrite.lmpdump2extxyz
```
````

````{py:function} _get_symbols_by_types(atoms: ase.atoms.Atoms)
:canonical: asext.io.readwrite._get_symbols_by_types

```{autodoc2-docstring} asext.io.readwrite._get_symbols_by_types
```
````
