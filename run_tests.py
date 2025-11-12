#!/usr/bin/env python3
"""
Script de utilidad para ejecutar tests del proyecto MLOps-GPO45.

Uso:
    python run_tests.py              # Ejecutar todos los tests
    python run_tests.py --quick      # Ejecutar solo tests rápidos
    python run_tests.py --coverage   # Ejecutar con reporte de cobertura
    python run_tests.py --integration # Ejecutar solo tests de integración
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description):
    """Ejecuta un comando del sistema y muestra el resultado."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}\n")
    print(f"Ejecutando: {' '.join(command)}\n")
    
    try:
        result = subprocess.run(command, check=True)
        print(f"\n✅ {description} - EXITOSO")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FALLÓ")
        return e.returncode
    except FileNotFoundError:
        print(f"\n⚠️  ERROR: pytest no está instalado")
        print("Instalar con: pip install pytest pytest-cov")
        return 1


def check_pytest_installed():
    """Verifica si pytest está instalado."""
    try:
        subprocess.run(["pytest", "--version"], 
                      capture_output=True, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutar tests del proyecto MLOps-GPO45",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python run_tests.py                    # Todos los tests
  python run_tests.py --quick            # Tests rápidos
  python run_tests.py --coverage         # Con cobertura de código
  python run_tests.py --integration      # Solo tests de integración
  python run_tests.py --unit             # Solo tests unitarios
  python run_tests.py --file cleaning    # Tests de un archivo específico
        """
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Ejecutar solo tests rápidos (sin integración)'
    )
    
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help='Ejecutar con reporte de cobertura de código'
    )
    
    parser.add_argument(
        '--integration', '-i',
        action='store_true',
        help='Ejecutar solo tests de integración'
    )
    
    parser.add_argument(
        '--unit', '-u',
        action='store_true',
        help='Ejecutar solo tests unitarios'
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Ejecutar tests de un archivo específico (ej: cleaning, preprocessing)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Modo verbose (más detalles)'
    )
    
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Ejecutar tests en paralelo (requiere pytest-xdist)'
    )
    
    args = parser.parse_args()
    
    # Verificar instalación de pytest
    if not check_pytest_installed():
        print("❌ ERROR: pytest no está instalado")
        print("\nPara instalar pytest y sus dependencias:")
        print("  pip install pytest pytest-cov pytest-xdist\n")
        return 1
    
    # Construir comando base
    command = ["pytest"]
    
    # Agregar opciones según argumentos
    if args.verbose:
        command.append("-v")
    else:
        command.append("-v")  # Siempre verbose por defecto
    
    if args.parallel:
        command.extend(["-n", "auto"])
    
    if args.coverage:
        command.extend([
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Filtrar por tipo de test
    if args.quick:
        command.extend([
            "tests/test_data_reader.py",
            "tests/test_data_explorer.py",
            "tests/test_data_cleaning.py",
        ])
        description = "Tests Rápidos"
    elif args.integration:
        command.append("tests/test_integration_pipeline.py")
        description = "Tests de Integración"
    elif args.unit:
        command.extend([
            "tests/test_data_reader.py",
            "tests/test_data_explorer.py",
            "tests/test_data_cleaning.py",
            "tests/test_data_preprocessing.py",
            "tests/test_data_analysis.py",
        ])
        description = "Tests Unitarios"
    elif args.file:
        test_file = f"tests/test_data_{args.file}.py"
        if not Path(test_file).exists():
            test_file = f"tests/test_{args.file}.py"
        command.append(test_file)
        description = f"Tests de {args.file}"
    else:
        command.append("tests/")
        description = "Suite Completa de Tests"
    
    # Ejecutar tests
    returncode = run_command(command, description)
    
    # Resumen final
    print(f"\n{'='*60}")
    if returncode == 0:
        print("✅ TODOS LOS TESTS PASARON")
        if args.coverage:
            print("\n📊 Reporte de cobertura generado en: htmlcov/index.html")
            print("   Abrir con: open htmlcov/index.html (Mac) o start htmlcov/index.html (Windows)")
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        print("\n💡 Sugerencias:")
        print("   - Revisar los errores arriba")
        print("   - Ejecutar tests individuales: pytest tests/test_<nombre>.py")
        print("   - Ver documentación: cat TESTING_README.md")
    print(f"{'='*60}\n")
    
    return returncode


if __name__ == "__main__":
    sys.exit(main())
