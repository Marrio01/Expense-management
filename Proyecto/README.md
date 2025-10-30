# Expense management

Aplicación que ayuda organizar las finanzas personales. El sistema combina los movimientos bancarios de BBVA (extraídos desde estados de cuenta en PDF) con los gastos en efectivo registrados manualmente, para ofrecer una visión completa de los ingresos y egresos.

La aplicación utiliza Machine Learning para clasificar automáticamente los gastos por categorías (comida, transporte, ocio, etc.) y predecir el gasto mensual futuro. Los resultados se muestran en un dashboard interactivo.

## Objetivo

Aplicación web que organice y prediga gastos personales combinando:
1. Movimientos bancarios
2. Gastos en efectivo
3. Modelo de Machine Learning para clasificar y predecir tendencias de consumo.
4. Dashboard interactivo

## Plan de trabajo

### Fase 1: Preparación del dataset
- Extraer movimientos desde PDFs de BBVA
- Limpieza y normalización de datos
- Creación de dataset para pruebas iniciales

### Fase 2: Registro de gastos en efectivo
- Formulario en la app para registrar efectivo
- Integración con los datos bancarios en un solo dataset

### Fase 3: Modelado ML
- Clasificación de gastos
- Predicción de gastos mensual futuro (Regresión lineal)


### Fase 4: Dashboard
- Desarrollo de las visualizaciones
* Distribución de gastos por categoría
* Línea del tiempo de gastos
* Predicción del próximo mes
* Filtros



