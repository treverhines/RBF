#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include "wrapped_code_7.h"

static PyMethodDef wrapper_module_7Methods[] = {
        {NULL, NULL, 0, NULL}
};

static void wrapped_281432003046400_ufunc(char **args, npy_intp *dimensions, npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    char *in1 = args[1];
    char *in2 = args[2];
    char *in3 = args[3];
    char *in4 = args[4];
    char *out0 = args[5];
    npy_intp in0_step = steps[0];
    npy_intp in1_step = steps[1];
    npy_intp in2_step = steps[2];
    npy_intp in3_step = steps[3];
    npy_intp in4_step = steps[4];
    npy_intp out0_step = steps[5];
    for (i = 0; i < n; i++) {
        *((double *)out0) = autofunc0(*(double *)in0, *(double *)in1, *(double *)in2, *(double *)in3, *(double *)in4);
        in0 += in0_step;
        in1 += in1_step;
        in2 += in2_step;
        in3 += in3_step;
        in4 += in4_step;
        out0 += out0_step;
    }
}
PyUFuncGenericFunction wrapped_281432003046400_funcs[1] = {&wrapped_281432003046400_ufunc};
static char wrapped_281432003046400_types[6] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};
static void *wrapped_281432003046400_data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "wrapper_module_7",
    NULL,
    -1,
    wrapper_module_7Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_wrapper_module_7(void)
{
    PyObject *m, *d;
    PyObject *ufunc0;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ufunc0 = PyUFunc_FromFuncAndData(wrapped_281432003046400_funcs, wrapped_281432003046400_data, wrapped_281432003046400_types, 1, 5, 1,
            PyUFunc_None, "wrapper_module_7", "Created in SymPy with Ufuncify", 0);
    PyDict_SetItemString(d, "wrapped_281432003046400", ufunc0);
    Py_DECREF(ufunc0);
    return m;
}
#else
PyMODINIT_FUNC initwrapper_module_7(void)
{
    PyObject *m, *d;
    PyObject *ufunc0;
    m = Py_InitModule("wrapper_module_7", wrapper_module_7Methods);
    if (m == NULL) {
        return;
    }
    import_array();
    import_umath();
    d = PyModule_GetDict(m);
    ufunc0 = PyUFunc_FromFuncAndData(wrapped_281432003046400_funcs, wrapped_281432003046400_data, wrapped_281432003046400_types, 1, 5, 1,
            PyUFunc_None, "wrapper_module_7", "Created in SymPy with Ufuncify", 0);
    PyDict_SetItemString(d, "wrapped_281432003046400", ufunc0);
    Py_DECREF(ufunc0);
}
#endif