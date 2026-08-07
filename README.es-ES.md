

# PharmaFormer: Aprendizaje por transferencia guiado por organoides para predecir la respuesta clínica a fármacos

**Autores**: Yuru Zhou, Quanhui Dai, Yanming Xu, Shuang Wu, Minzhang Cheng,* y Bing Zhao,*

## Descripción general

PharmaFormer es un modelo de aprendizaje profundo basado en **Transformers** diseñado para predecir la respuesta clínica a fármacos integrando perfiles de expresión génica y estructuras moleculares de los fármacos. Este modelo combina datos de organoides tumorales derivados de pacientes con conjuntos de datos farmacogenómicos a gran escala de líneas celulares, aprovechando un enfoque de aprendizaje por transferencia para mejorar la precisión de las predicciones.

<p align="center">
  <img src="Schematic of the PharmaFormer model.jpg" alt="PharmaFormer Architecture" width="500"/>
</p>

## Características principales

1. **Aprendizaje por transferencia sin fisuras**: PharmaFormer aprovecha un modelo preentrenado desarrollado con datos extensos de líneas celulares (87.596 pares célula-fármaco) del conjunto de datos GDSC2 y lo ajusta fino con datos específicos de organoides, mejorando su capacidad para predecir la respuesta a fármacos de los pacientes con alta precisión.
2. **Aprovechamiento de información de organoides**: Al utilizar la fidelidad biomimética de los organoides, que reflejan de cerca las respuestas clínicas a fármacos, PharmaFormer destaca el papel crucial de los datos de organoides en el avance de los modelos de predicción de respuesta a fármacos.
3. **Rendimiento predictivo mejorado**: PharmaFormer sobresale en la predicción de respuesta a fármacos, particularmente para pacientes con tumores clínicos, proporcionando predicciones robustas y precisas que apoyan los esfuerzos de oncología personalizada y medicina de precisión.

## Instalación
```bash
git clone https://github.com/zhouyuru1205/PharmaFormer.git
cd PharmaFormer
pip install -r requirements.txt
```

## Agradecimientos

Agradecemos a Pascal Iversen sus comentarios constructivos durante el benchmarking de PharmaFormer. El código de evaluación se ha actualizado para alinearse con la metodología del manuscrito.

## Citas

Si considera que PharmaFormer es útil para su investigación, cite:

> Zhou, Y., Dai, Q., Xu, Y. et al. PharmaFormer predicts clinical drug responses through transfer learning guided by patient derived organoid. *npj Precis. Onc.* (2025).
