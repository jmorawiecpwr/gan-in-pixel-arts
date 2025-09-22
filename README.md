# Generowanie póz do pixel-artów
Nasz projekt polega na stworzeniu modelu, który dostając zdjęcie (pixel-art) postaci, jest w stanie wygenerować zdjęcie tej samej postaci ale w nowej pozie (np. ze stojącej w siedzącą)
W tym celu wykorzystamy sieć GAN, która pobiera dane z własnoręcznie stworzonego datasetu pixel-artów (około milion zdjęć 48x48 pixeli). Dataset powstał z wykorzystaniem generatora tego typu postaci.
### Wykorzystanie
Nasz projekt ma na celu automatyzację tworzenia różnych wariantów postaci co można wykorzystać w retro grach wideo po gry typu D&D
## Tworzenie projektu:
### Krok 1 - Dataset
Dane pozyskaliśmy przy pomocy strony [RPG maker](https://www.rpgmakerweb.com), gdzie udało nam się wygenerować w przybliżeniu milion pixelowych postaci które zostaną przetworzone przez model.
Po wygenerowaniu wszystkich obrazów, zostały one przetworzone aby każde zdjęcie miało 48x48 pixeli. Całość została podzielona na foldery po +/- 16000 obrazków.
### Krok 2 - ładowanie danych do modelu
stworzyliśmy dataloader, który normalizuje wsszystkie zdjęcia do zakresu [-1,1], i wyrzuca je w formie batchy po 64 zdjęcia do modelu, dateset jest podzielony na foldery, gdzie każdy folder to osobna poza, w każdym folderze jest 15922 postaci ustawionych w tej samej kolejności co ułatwia trening modelu.
### Krok 3 - Model
Docelowo chcemy przekształcać ludzika w 1 pozie do 4 innych póz, gdzie użytkownik wybiera jaką chce pozę. Wykorzystamy do tego model GAN typu pix2pix który uczy się 2 wybranych póz aby móc potem transformować jedną pozę w drugą, przez co wystarczy zrobic 4 takie modele dla 4 różych póz (poza startowa jest taka sama dla każdego modelu)