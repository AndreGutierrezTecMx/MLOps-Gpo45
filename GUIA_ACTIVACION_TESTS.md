# 🔧 Guía de Activación de Tests - MLOps-GPO45

## ⚠️ IMPORTANTE: Tests Como Templates

Los archivos de test creados contienen **templates comentados** que necesitan ser activados. Esto se hizo intencionalmente para:

1. ✅ Proporcionar estructura completa de tests
2. ✅ Evitar errores de imports hasta que estén listos
3. ✅ Permitir activación gradual según avance del proyecto

---

## 📋 Pasos para Activar los Tests

### Paso 1: Instalar Dependencias de Testing

```bash
pip install pytest pytest-cov pytest-xdist pytest-timeout
```

### Paso 2: Verificar Estructura del Proyecto

Asegúrate de que tu proyecto tenga esta estructura:

```
MLOps-GPO45/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_reader.py
│   │   ├── data_explorer.py
│   │   ├── data_cleaning.py
│   │   ├── data_preprocessing.py
│   │   └── data_analysis.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py
│   └── versioning/
│       ├── __init__.py
│       └── version_tracker.py
├── tests/                    # ← Carpeta de tests (nueva)
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data_reader.py
│   ├── test_data_explorer.py
│   ├── test_data_cleaning.py
│   ├── test_data_preprocessing.py
│   ├── test_data_analysis.py
│   └── test_integration_pipeline.py
├── pytest.ini
├── run_tests.py
└── TESTING_README.md
```

### Paso 3: Activar Tests Gradualmente

#### 🟢 Opción A: Activar UN test a la vez (Recomendado para empezar)

1. Abre `tests/test_data_cleaning.py`
2. Encuentra un test simple, por ejemplo `test_initialization_creates_copy`
3. Descomenta las líneas del test:

```python
# ANTES (comentado):
def test_initialization_creates_copy(self, sample_dataframe, mock_version_tracker):
    """Test que la inicialización crea una copia del DataFrame."""
    # from src.data.data_cleaning import DataCleaning
    # cleaner = DataCleaning(sample_dataframe, mock_version_tracker)
    # assert cleaner.df_clean is not None
    pass

# DESPUÉS (descomentado):
def test_initialization_creates_copy(self, sample_dataframe, mock_version_tracker):
    """Test que la inicialización crea una copia del DataFrame."""
    from src.data.data_cleaning import DataCleaning
    cleaner = DataCleaning(sample_dataframe, mock_version_tracker)
    assert cleaner.df_clean is not None
    assert isinstance(cleaner.df_clean, pd.DataFrame)
    assert cleaner.df_clean is not sample_dataframe
```

4. Ejecutar ese test específico:
```bash
pytest tests/test_data_cleaning.py::TestDataCleaningInitialization::test_initialization_creates_copy -v
```

#### 🟡 Opción B: Activar toda una CLASE de tests

1. Abre el archivo de test (ej: `test_data_cleaning.py`)
2. Descomenta todos los tests de una clase (ej: `TestDataCleaningInitialization`)
3. Ejecutar:
```bash
pytest tests/test_data_cleaning.py::TestDataCleaningInitialization -v
```

#### 🔴 Opción C: Activar TODO un archivo de tests

1. Abre el archivo completo
2. Descomenta TODOS los imports y código de tests
3. Ejecutar:
```bash
pytest tests/test_data_cleaning.py -v
```

---

## 🎯 Orden Recomendado de Activación

### Nivel 1: Tests Básicos (Comenzar aquí) ✅

```bash
# 1. Primero activar tests de inicialización
tests/test_data_cleaning.py::TestDataCleaningInitialization
tests/test_data_preprocessing.py::TestPreprocessorInitialization

# 2. Luego activar tests de funciones individuales
tests/test_data_cleaning.py::TestConvertDataTypes
tests/test_data_cleaning.py::TestRemoveDuplicates
```

### Nivel 2: Tests Intermedios

```bash
# 3. Activar tests más complejos
tests/test_data_cleaning.py::TestHandleMissingValues
tests/test_data_cleaning.py::TestHandleOutliers
tests/test_data_preprocessing.py::TestBuildXY
```

### Nivel 3: Tests Avanzados

```bash
# 4. Activar tests de integración
tests/test_integration_pipeline.py::TestCompleteDataPipeline
tests/test_integration_pipeline.py::TestEndToEndPipeline
```

---

## 🔍 Ejemplo Completo: Activar Test de DataCleaning

### Archivo Original (comentado):

```python
def test_convert_string_to_numeric(self, mock_version_tracker):
    """Test conversión de strings a numéricos."""
    df = pd.DataFrame({
        'url': ['http://example.com/1', 'http://example.com/2'],
        'shares': ['1000', '2000'],
        'n_tokens': ['10', '20']
    })
    
    # from src.data.data_cleaning import DataCleaning
    # cleaner = DataCleaning(df, mock_version_tracker)
    # cleaner.convert_data_types()
    
    # assert pd.api.types.is_numeric_dtype(cleaner.df_clean['shares'])
    # assert pd.api.types.is_numeric_dtype(cleaner.df_clean['n_tokens'])
    # assert cleaner.df_clean['url'].dtype == object
    pass
```

### Archivo Activado (descomentado):

```python
def test_convert_string_to_numeric(self, mock_version_tracker):
    """Test conversión de strings a numéricos."""
    df = pd.DataFrame({
        'url': ['http://example.com/1', 'http://example.com/2'],
        'shares': ['1000', '2000'],
        'n_tokens': ['10', '20']
    })
    
    from src.data.data_cleaning import DataCleaning
    cleaner = DataCleaning(df, mock_version_tracker)
    cleaner.convert_data_types()
    
    assert pd.api.types.is_numeric_dtype(cleaner.df_clean['shares'])
    assert pd.api.types.is_numeric_dtype(cleaner.df_clean['n_tokens'])
    assert cleaner.df_clean['url'].dtype == object
    # Remover el 'pass' al descomentar el código
```

---

## 🚀 Quick Start - Primeros 5 Minutos

```bash
# 1. Instalar dependencias
pip install pytest pytest-cov

# 2. Verificar que pytest funciona
pytest --version

# 3. Ejecutar tests (aunque estén comentados, no debe fallar)
pytest tests/ -v

# 4. Activar tu primer test
# Edita tests/test_data_cleaning.py
# Descomenta un test simple
# Ejecuta: pytest tests/test_data_cleaning.py -v

# 5. Ver cobertura
pytest --cov=src --cov-report=term
```

---

## 🛠️ Troubleshooting Común

### Error: "No module named 'src'"

**Solución 1:** Agregar al PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

**Solución 2:** Instalar en modo desarrollo
```bash
pip install -e .
```

**Solución 3:** Agregar al inicio de conftest.py
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Error: "fixture 'sample_dataframe' not found"

**Causa:** `conftest.py` no está en el lugar correcto.

**Solución:** Verificar que `tests/conftest.py` existe.

### Error: Tests pasan pero no hacen nada

**Causa:** Los tests tienen `pass` y el código está comentado.

**Solución:** Descomentar el código del test (ver ejemplos arriba).

---

## 📊 Verificar Progreso

### Ver cuántos tests están activos:
```bash
# Total de tests
pytest --collect-only | grep "test session starts"

# Tests por archivo
pytest tests/test_data_cleaning.py --collect-only
```

### Ver cobertura de código:
```bash
pytest --cov=src --cov-report=term-missing
```

---

## ✅ Checklist de Activación

- [ ] Dependencias de testing instaladas (`pip install pytest pytest-cov`)
- [ ] Estructura de carpetas correcta (`tests/` en raíz del proyecto)
- [ ] `conftest.py` con fixtures disponibles
- [ ] Primer test descomentado y ejecutándose
- [ ] Imports funcionando correctamente
- [ ] Tests pasando (aunque sea 1)
- [ ] Cobertura de código verificada

---

## 📚 Próximos Pasos

1. **Activar tests progresivamente** - No intentes activar todo de una vez
2. **Ejecutar tests frecuentemente** - Después de cada cambio al código
3. **Mantener cobertura alta** - Objetivo: >80% en módulos críticos
4. **Agregar nuevos tests** - Para funcionalidad nueva
5. **Integrar en CI/CD** - Automatizar ejecución de tests

---

## 💡 Tips Profesionales

### Desarrollo iterativo:
```bash
# Durante desarrollo, ejecuta tests continuamente
pytest -x -v  # Detiene en primer fallo

# Cuando algo falla, ejecuta solo ese test
pytest tests/test_data_cleaning.py::TestClass::test_method -v

# Usa --lf para re-ejecutar últimos fallos
pytest --lf
```

### Debugging:
```bash
# Con prints visibles
pytest -s

# Con debugger
pytest --pdb

# Modo verbose extremo
pytest -vv
```

---

## 📞 Soporte

Si tienes problemas:

1. Revisa `TESTING_README.md` para documentación completa
2. Verifica que la estructura de archivos es correcta
3. Asegúrate de que los imports están bien
4. Ejecuta `python run_tests.py --help` para ver opciones

---

**¡Listo para empezar!** 🎉

Comienza activando un test simple y ve progresando gradualmente.
