# {py:mod}`asext.cell`

```{py:module} asext.cell
```

```{autodoc2-docstring} asext.cell
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AseCell <asext.cell.AseCell>`
  - ```{autodoc2-docstring} asext.cell.AseCell
    :summary:
    ```
* - {py:obj}`CellTransform <asext.cell.CellTransform>`
  - ```{autodoc2-docstring} asext.cell.CellTransform
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`make_upper_triangular_cell <asext.cell.make_upper_triangular_cell>`
  - ```{autodoc2-docstring} asext.cell.make_upper_triangular_cell
    :summary:
    ```
* - {py:obj}`make_lower_triangular_cell <asext.cell.make_lower_triangular_cell>`
  - ```{autodoc2-docstring} asext.cell.make_lower_triangular_cell
    :summary:
    ```
* - {py:obj}`make_triangular_cell_extxyz <asext.cell.make_triangular_cell_extxyz>`
  - ```{autodoc2-docstring} asext.cell.make_triangular_cell_extxyz
    :summary:
    ```
* - {py:obj}`_polar_rotation <asext.cell._polar_rotation>`
  - ```{autodoc2-docstring} asext.cell._polar_rotation
    :summary:
    ```
* - {py:obj}`rotate_struct_property <asext.cell.rotate_struct_property>`
  - ```{autodoc2-docstring} asext.cell.rotate_struct_property
    :summary:
    ```
````

### API

`````{py:class} AseCell(array: numpy.ndarray)
:canonical: asext.cell.AseCell

Bases: {py:obj}`ase.cell.Cell`

```{autodoc2-docstring} asext.cell.AseCell
```

```{rubric} Initialization
```

```{autodoc2-docstring} asext.cell.AseCell.__init__
```

````{py:method} lower_triangular_form() -> tuple[ase.cell.Cell, numpy.ndarray]
:canonical: asext.cell.AseCell.lower_triangular_form

```{autodoc2-docstring} asext.cell.AseCell.lower_triangular_form
```

````

````{py:method} upper_triangular_form() -> tuple[ase.cell.Cell, numpy.ndarray]
:canonical: asext.cell.AseCell.upper_triangular_form

```{autodoc2-docstring} asext.cell.AseCell.upper_triangular_form
```

````

`````

````{py:function} make_upper_triangular_cell(atoms: ase.atoms.Atoms, zero_tol: float = 1e-12) -> ase.atoms.Atoms
:canonical: asext.cell.make_upper_triangular_cell

```{autodoc2-docstring} asext.cell.make_upper_triangular_cell
```
````

````{py:function} make_lower_triangular_cell(atoms: ase.atoms.Atoms, zero_tol: float = 1e-12) -> ase.atoms.Atoms
:canonical: asext.cell.make_lower_triangular_cell

```{autodoc2-docstring} asext.cell.make_lower_triangular_cell
```
````

````{py:function} make_triangular_cell_extxyz(extxyz_file: str, form: str = 'lower') -> None
:canonical: asext.cell.make_triangular_cell_extxyz

```{autodoc2-docstring} asext.cell.make_triangular_cell_extxyz
```
````

`````{py:class} CellTransform(old_cell: numpy.ndarray, new_cell: numpy.ndarray, pure_rotation: bool = True)
:canonical: asext.cell.CellTransform

```{autodoc2-docstring} asext.cell.CellTransform
```

```{rubric} Initialization
```

```{autodoc2-docstring} asext.cell.CellTransform.__init__
```

````{py:method} vectors_forward(vec: numpy.ndarray) -> numpy.ndarray
:canonical: asext.cell.CellTransform.vectors_forward

```{autodoc2-docstring} asext.cell.CellTransform.vectors_forward
```

````

````{py:method} vectors_backward(vec: numpy.ndarray) -> numpy.ndarray
:canonical: asext.cell.CellTransform.vectors_backward

```{autodoc2-docstring} asext.cell.CellTransform.vectors_backward
```

````

````{py:method} tensor_forward(tensor: numpy.ndarray) -> numpy.ndarray
:canonical: asext.cell.CellTransform.tensor_forward

```{autodoc2-docstring} asext.cell.CellTransform.tensor_forward
```

````

````{py:method} tensor_backward(tensor: numpy.ndarray) -> numpy.ndarray
:canonical: asext.cell.CellTransform.tensor_backward

```{autodoc2-docstring} asext.cell.CellTransform.tensor_backward
```

````

`````

````{py:function} _polar_rotation(A: numpy.ndarray) -> numpy.ndarray
:canonical: asext.cell._polar_rotation

```{autodoc2-docstring} asext.cell._polar_rotation
```
````

````{py:function} rotate_struct_property(struct: ase.atoms.Atoms, new_cell: numpy.ndarray, wrap: bool = False, custom_vector_props: list[str] | None = None, custom_tensor_props: list[str] | None = None) -> ase.atoms.Atoms
:canonical: asext.cell.rotate_struct_property

```{autodoc2-docstring} asext.cell.rotate_struct_property
```
````
