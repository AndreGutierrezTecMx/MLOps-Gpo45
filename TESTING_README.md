# 🧪 Testing Suite - MLOps-GPO45

Suite completa de pruebas automatizadas para validar componentes críticos del proyecto MLOps-GPO45.

## 📋 Tabla de Contenidos

- [Estructura de Tests](#estructura-de-tests)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Ejecución de Tests](#ejecución-de-tests)
- [Tipos de Tests](#tipos-de-tests)
- [Cobertura de Código](#cobertura-de-código)
- [Mejores Prácticas](#mejores-prácticas)

---

## 🗂️ Estructura de Tests

```
tests/
├── __init__.py                      # Inicialización del paquete de tests
├── conftest.py                      # Fixtures compartidos entre tests
├── test_data_reader.py              # Tests para DataReader
├── test_data_explorer.py            # Tests para DataExplorer
├── test_data_cleaning.py            # Tests para DataCleaning ⭐
├── test_data_preprocessing.py       # Tests para Preprocessor ⭐
├── test_data_analysis.py            # Tests para DataAnalysis
└── test_integration_pipeline.py    # Tests de integración end-to-end ⭐
```

**⭐ = Tests prioritarios**

---

## 📦 Requisitos

### Dependencias Base
```bash
pytest>=7.4.0
pytest-cov>=4.1.0           # Para cobertura de código
pytest-xdist>=3.3.1         # Para ejecución paralela (opcional)
pytest-timeout>=2.1.0       # Para timeout de tests (opcional)
```

### Instalar Dependencias de Testing
```bash
pip install pytest pytest-cov pytest-xdist pytest-timeout
```

---

## 🚀 Ejecución de Tests

### Comando Básico
Ejecutar todos los tests:
```bash
pytest
```

### Comando Recomendado (Detallado)
```bash
pytest -v
```

### Modo Silencioso
```bash
pytest -q
```

### Ejecutar Tests Específicos

**Por archivo:**
```bash
pytest tests/test_data_cleaning.py
pytest tests/test_data_preprocessing.py
pytest tests/test_integration_pipeline.py
```

**Por clase:**
```bash
pytest tests/test_data_cleaning.py::TestDataCleaningInitialization
pytest tests/test_data_preprocessing.py::TestPreprocessorInitialization
```

**Por función específica:**
```bash
pytest tests/test_data_cleaning.py::TestConvertDataTypes::test_convert_string_to_numeric
```

**Por patrón de nombre:**
```bash
pytest -k "cleaning"        # Ejecuta tests con "cleaning" en el nombre
pytest -k "test_initialization"  # Ejecuta tests de inicialización
```

### Ejecutar por Marcadores

**Tests unitarios:**
```bash
pytest -m unit
```

**Tests de integración:**
```bash
pytest -m integration
```

**Tests rápidos (smoke tests):**
```bash
pytest -m smoke
```

### Opciones Útiles

**Detener en el primer fallo:**
```bash
pytest -x
```

**Ejecutar último test fallido:**
```bash
pytest --lf
```

**Ejecutar tests fallidos primero:**
```bash
pytest --ff
```

**Mostrar output completo (print statements):**
```bash
pytest -s
```

**Ejecutar en paralelo (requiere pytest-xdist):**
```bash
pytest -n auto      # Usa todos los cores disponibles
pytest -n 4         # Usa 4 workers
```

**Con timeout:**
```bash
pytest --timeout=60
```

---

## 🔍 Tipos de Tests

### 1. Tests Unitarios
Validan funciones y métodos individuales de forma aislada.

**Ejemplos:**
- `test_data_cleaning.py`: Tests para cada método de DataCleaning
- `test_data_preprocessing.py`: Tests para transformaciones y splits
- `test_data_reader.py`: Tests para lectura de archivos

**Ejecutar solo tests unitarios:**
```bash
pytest tests/test_data_cleaning.py tests/test_data_preprocessing.py
```

### 2. Tests de Integración
Validan el flujo completo entre múltiples componentes.

**Archivo principal:**
- `test_integration_pipeline.py`: Tests end-to-end del pipeline completo

**Ejecutar tests de integración:**
```bash
pytest tests/test_integration_pipeline.py
```

### 3. Tests de Fixtures
Los fixtures en `conftest.py` proveen datos de prueba reutilizables:
- `sample_dataframe`: DataFrame limpio de prueba
- `sample_dataframe_with_nulls`: DataFrame con valores faltantes
- `sample_dataframe_with_duplicates`: DataFrame con duplicados
- `sample_dataframe_with_outliers`: DataFrame con valores atípicos
- `mock_version_tracker`: Mock del VersionTracker
- `temp_csv_file`: Archivo CSV temporal

---

## 📊 Cobertura de Código

### Generar Reporte de Cobertura

**HTML (recomendado):**
```bash
pytest --cov=src --cov-report=html
```
Luego abrir `htmlcov/index.html` en el navegador.

**Terminal:**
```bash
pytest --cov=src --cov-report=term-missing
```

**Cobertura por módulo:**
```bash
pytest --cov=src.data --cov-report=term
```

### Interpretación de Cobertura

| Cobertura | Calificación | Acción Recomendada |
|-----------|--------------|-------------------|
| 90-100%   | Excelente ✅ | Mantener |
| 75-89%    | Buena 👍     | Mejorar áreas críticas |
| 60-74%    | Aceptable ⚠️ | Agregar más tests |
| < 60%     | Insuficiente ❌ | Prioridad alta |

---

## 📝 Mejores Prácticas

### 1. Nomenclatura
- Archivos: `test_<nombre_modulo>.py`
- Clases: `Test<NombreClase>`
- Funciones: `test_<descripcion_comportamiento>`

### 2. Estructura de Tests
```python
def test_nombre_descriptivo(fixture1, fixture2):
    """Docstring explicando qué se está testeando."""
    # Arrange (Preparar)
    data = prepare_test_data()
    
    # Act (Actuar)
    result = function_under_test(data)
    
    # Assert (Verificar)
    assert result == expected_value
```

### 3. Uso de Fixtures
Usar fixtures de `conftest.py` en lugar de crear datos en cada test:
```python
def test_with_fixture(sample_dataframe, mock_version_tracker):
    cleaner = DataCleaning(sample_dataframe, mock_version_tracker)
    # ...
```

### 4. Tests Parametrizados
Para probar múltiples casos:
```python
@pytest.mark.parametrize("strategy", ['drop', 'mean', 'median'])
def test_missing_values(strategy, sample_dataframe_with_nulls):
    # Test con diferentes estrategias
    pass
```

### 5. Mocks
Usar mocks para evitar dependencias externas:
```python
@patch('dvc.api.get_url')
def test_with_mock(mock_get_url):
    mock_get_url.return_value = "fake_url"
    # ...
```

---

## 🎯 Comandos Más Usados

### Desarrollo Diario
```bash
# Tests rápidos durante desarrollo
pytest -x -v

# Tests con cobertura
pytest --cov=src --cov-report=term-missing

# Re-ejecutar últimos fallos
pytest --lf
```

### Pre-Commit / CI
```bash
# Suite completa con reporte
pytest -v --cov=src --cov-report=html --cov-report=term

# Suite completa en paralelo
pytest -n auto -v
```

### Debugging
```bash
# Con output completo
pytest -s -v

# Con debugger (pdb)
pytest --pdb

# Verbose extremo
pytest -vv
```

---

## 🐛 Troubleshooting

### Problema: "No module named 'src'"
**Solución:**
```bash
# Agregar el directorio raíz al PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# O instalar el paquete en modo desarrollo
pip install -e .
```

### Problema: Tests comentados no se ejecutan
**Causa:** Los tests están comentados (usando `pass`).

**Solución:** Descomentar las líneas del test que desees ejecutar.

### Problema: ImportError en fixtures
**Solución:** Asegurarse de que `conftest.py` esté en el directorio `tests/`.

### Problema: "fixture not found"
**Solución:** Verificar que la fixture esté definida en `conftest.py` o en el mismo archivo de test.

---

## 📈 Siguientes Pasos

1. **Descomentar Tests**: Los tests actuales están como templates. Descomentar los imports y el código de test.

2. **Ejecutar Suite Básica**:
   ```bash
   pytest -v
   ```

3. **Agregar Tests Específicos**: Agregar tests adicionales según necesidades del proyecto.

4. **Integrar en CI/CD**: Agregar ejecución de tests al pipeline de CI/CD:
   ```yaml
   # .github/workflows/tests.yml
   - name: Run tests
     run: pytest --cov=src --cov-report=xml
   ```

5. **Monitorear Cobertura**: Mantener cobertura > 80% para componentes críticos.

---

## 🔗 Referencias

- [Documentación de pytest](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

## ✅ Checklist de Testing

- [ ] Todos los tests pasan localmente
- [ ] Cobertura > 80% en módulos críticos
- [ ] Tests de integración ejecutados exitosamente
- [ ] Sin warnings o deprecations
- [ ] Documentación de tests actualizada
- [ ] Tests agregados al CI/CD

---

**Última actualización:** Noviembre 2025  
**Autor:** Equipo MLOps-GPO45  
**Contacto:** [Tu equipo]
