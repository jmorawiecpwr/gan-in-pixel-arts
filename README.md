# Generowanie póz do pixel-artów
Nasz projekt polega na stworzeniu modelu, który dostając zdjęcie (pixel-art) postaci, jest w stanie wygenerować zdjęcie tej samej postaci ale w nowej pozie (np. ze stojącej w siedzącą)
W tym celu wykorzystamy sieć GAN, która pobiera dane z własnoręcznie stworzonego datasetu pixel-artów (około milion zdjęć 48x48 pixeli). Dataset powstał z wykorzystaniem generatora tego typu postaci.
### Wykorzystanie
Nasz projekt ma na celu automatyzację tworzenia różnych wariantów postaci co można wykorzystać w retro grach wideo po gry typu D&D
## Tworzenie projektu:
### Krok 1 - Dataset
Dane pozyskaliśmy przy pomocy strony [RPG maker](https://www.rpgmakerweb.com), gdzie udało nam się wygenerować w przybliżeniu milion pixelowych postaci które zostaną przetworzone przez model.
Po wygenerowaniu wszystkich obrazów, zostały one przetworzone aby każde zdjęcie miało 48x48 pixeli. Całość została podzielona na foldery po +/- 16000 obrazków.
