# {py:mod}`asext.struct`

```{py:module} asext.struct
```

```{autodoc2-docstring} asext.struct
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`strain_struct <asext.struct.strain_struct>`
  - ```{autodoc2-docstring} asext.struct.strain_struct
    :summary:
    ```
* - {py:obj}`perturb_struct <asext.struct.perturb_struct>`
  - ```{autodoc2-docstring} asext.struct.perturb_struct
    :summary:
    ```
* - {py:obj}`slice_struct <asext.struct.slice_struct>`
  - ```{autodoc2-docstring} asext.struct.slice_struct
    :summary:
    ```
* - {py:obj}`align_struct_min_pos <asext.struct.align_struct_min_pos>`
  - ```{autodoc2-docstring} asext.struct.align_struct_min_pos
    :summary:
    ```
* - {py:obj}`set_vacuum <asext.struct.set_vacuum>`
  - ```{autodoc2-docstring} asext.struct.set_vacuum
    :summary:
    ```
* - {py:obj}`check_bad_box_extxyz <asext.struct.check_bad_box_extxyz>`
  - ```{autodoc2-docstring} asext.struct.check_bad_box_extxyz
    :summary:
    ```
* - {py:obj}`check_bad_box <asext.struct.check_bad_box>`
  - ```{autodoc2-docstring} asext.struct.check_bad_box
    :summary:
    ```
````

### API

````{py:function} strain_struct(input_struct: ase.atoms.Atoms, strains: list[float] = [0.0, 0, 0]) -> ase.atoms.Atoms
:canonical: asext.struct.strain_struct

```{autodoc2-docstring} asext.struct.strain_struct
```
````

````{py:function} perturb_struct(struct: ase.atoms.Atoms, std_disp: float) -> ase.atoms.Atoms
:canonical: asext.struct.perturb_struct

```{autodoc2-docstring} asext.struct.perturb_struct
```
````

````{py:function} slice_struct(struct_in: ase.atoms.Atoms, slice_num=(1, 1, 1), tol=1e-05) -> ase.atoms.Atoms
:canonical: asext.struct.slice_struct

```{autodoc2-docstring} asext.struct.slice_struct
```
````

````{py:function} align_struct_min_pos(struct: ase.atoms.Atoms) -> ase.atoms.Atoms
:canonical: asext.struct.align_struct_min_pos

```{autodoc2-docstring} asext.struct.align_struct_min_pos
```
````

````{py:function} set_vacuum(input_struct: ase.atoms.Atoms, distances: list = [0.0, 0.0, 0.0]) -> ase.atoms.Atoms
:canonical: asext.struct.set_vacuum

```{autodoc2-docstring} asext.struct.set_vacuum
```
````

````{py:function} check_bad_box_extxyz(extxyz_file: str, criteria: dict = {'length_ratio': 100, 'wrap_ratio': 0.5, 'tilt_ratio': 0.5}) -> bool
:canonical: asext.struct.check_bad_box_extxyz

```{autodoc2-docstring} asext.struct.check_bad_box_extxyz
```
````

````{py:function} check_bad_box(struct: ase.atoms.Atoms, criteria: dict = {'length_ratio': 20, 'wrap_ratio': 0.5, 'tilt_ratio': 0.5}) -> bool
:canonical: asext.struct.check_bad_box

```{autodoc2-docstring} asext.struct.check_bad_box
```
````
