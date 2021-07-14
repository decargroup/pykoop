# pykoop

Koopman operator identification library in Python.

## To do

### Pipeline

- [ ] Add post-processing step for normalization of quaternions etc.
- [ ] Merge LMI methods with new abstract base class
- [ ] Fix examples (Must merge LMI methods first)
- [ ] Handle methods that don't save a `coef_`?

### LMI

- [ ] Make `_add_regularizer` return a problem instead of acting in-place?
- [ ] Remove constraint-adding functions and wrap them inside an inherited
  `_base_problem`? Might cause problems when trying to mix regularizers.
- [ ] Rename `_base_problem` to `_get_base_problem`
- [ ] Rename `U_` to `coef_`
- [ ] Add nonhomog test case
- [ ] Adjust fit and predict testing to avoid skipped tests. Parametrize
  arguments for fixtures?
