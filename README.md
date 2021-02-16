# pykoop

Koopman operator identification library in Python.

## To do

- [ ] Make `_add_regularizer` return a problem instead of acting in-place?
- [ ] Remove constraint-adding functions and wrap them inside an inherited
  `_base_problem`? Might cause problems when trying to mix regularizers.
- [ ] Rename `_base_problem` to `_get_base_problem`
- [ ] Rename `U_` to `coef_`
- [ ] Add nonhomog test case
- [ ] Adjust fit and predict testing to avoid skipped tests
