# Chrome Dino - Reinforcement Learning Project

Este proyecto implementa y compara diferentes estrategias de **Aprendizaje por Refuerzo (Reinforcement Learning)** para resolver el juego del Dinosaurio de Chrome. Se utiliza la librería **Stable-Baselines3** y un entorno personalizado compatible con **Gymnasium**.

El código base del juego es una adaptación del trabajo original de [maxrohowsky](https://github.com/MaxRohowsky/chrome-dinosaur).

## Descripción

El objetivo del proyecto es entrenar a un agente autónomo capaz de saltar cactus y esquivar pájaros indefinidamente. Se exploran las siguientes técnicas:

* **Classic Learning:** Entrenamiento estándar utilizando algoritmos **DQN** y **PPO** con distintas funciones de recompensa.
* **Curriculum Learning:** Entrenamiento progresivo por fases (Cactus pequeños $\to$ Entorno completo $\to$ Pájaros a distintas alturas).
* **Imitation Learning:** Uso de **Behavioral Cloning** y refinamiento con DQN a partir de demostraciones humanas.
