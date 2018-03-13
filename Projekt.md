1. SI
   - Będzie się składała z kilku warstw *(o nich później)*, z nich zbudujemy sieć typu feed forward, która na wejściu otrzyma wektor ze stanem gry, a na wyjściu otrzymamy wektor z wartościami Q dla poszczególnych akcji
   - musi posiadać metodę w stylu *wybierz_akcję(stan)*, która zwróci ID akcji o najwyższej wartości Q
   - musi posiadać strukturę danych na *pamięć*, dane przechowywane w pamięci to będzie krotka: (stan, akcja, nagroda, stan_wynikowy, czy_koniec) *jak w chyba każdej implementacji q-learninqu*
   - metoda *zapamiętaj(paczka_informacjami_do_zapamiętania)*
   - musi posiadać metodę w stylu *ucz_się()*, która wybierze losową próbkę z *pamięci* i na jej podstawie wyznaczy dane/daną potrzebne do wywołania metody aktualizacji wag w sieci neruonowej *(poczytać o gradient descent i realizacji tego w deep q learning)*
   - Chyba powinno też mieć jakąś metodę do serializacji, przynajmniej taką co wywoła podobną metodę na sieci neuronowej i zapisze nasze ciężko wyliczone wagi do pliku
2. Sieć neuronowa
   - niech to będzie taki kontener na warstwy sieci, które połączymy za zasadzie listy, wtedy realizacja przepływu danych przez sieć to będzie kwestia wywołania jednej metody na początku tej listy, a realizacja aktualizacji wag - wywołanie odpowiedniej metody na końcu tej listy
   - konstruktor powinien pobierać ilości neruonów na poszczególnych warstwach *(póki co niech to będą 3 liczby, później ewentualnie rozszerzymy)*
   - metoda *myśl(dane_wejściowe)* (czy coś w ten deseń),  wykona operację przepływu na pierwszej warstwie i zwróci rezultat przepływu danych przez całą sieć
   - metoda do aktualizacji wag
3. Warstwa sieci neuronowej
   - Z tego bym zrobił taką otoczkę na całą matematykę co stoi za naszą sięcią neuronową
   - Tu będzie konieczne użycie NumPy
   - Inicjalizaja = utworzenie macierzy o zadanych wymiarach i wypełnienie jej losowymi wagami
   - realizacja funkcji aktywacji *(z tego co wyczytałem będziemy korzystać z liniowej aktywacji na ostatniej warstwie i z ReLu na pozostałych)*
   - Metoda aktualizacji wag, wykonanie aktualizacji na warstwie poprzedniej
   - Matematyczna realizacja przepływu informacji *(mnożenie macierzy wejściowej przez macierz wag i nałożenie aktywacji, wysłanie wyniku do następnej warstwy, zwrócenie wyniku z następnej warstwy)*
4. Abstrakcja gry
   - to jest tak naprawdę sztuka dla sztuki, parę linii kodu, ale nie wiemy czy nie będziemy rozszerzać projektu o inne gry, a tak będą prawilnie dziedziczyć po klasie Game i będą musiały realizować pewne metody
   - *wykonaj(akcja)* - powinno zwrócić nowy stan, nagrodę za wybraną akcję
   - *rysuj()* - opcjonalne wywołanie, gdybyśmy chcieli obserwować przebieg rozgrywki
   - *start()* - niech rozpocznie grę od nowa i zwróci stan początkowy
   - *czy_koniec()* do zapisania informacji o końcu gry w pamięci SI
   - jakaś podstawowa inicjalizacja, w której można zdefiniować długość rozgrywki?
   - w sumie nie wiem czy coś jeszcze jest potrzebne
5. Gra (pong)
   - No i tu jest chyba wszystko jasne, najlepiej wziąć gotowego ponga, dostosować go do naszych wymagań i do dzieła!
   - Tu też chyba powinnien być zrealizowany jakiś bot, który działał by w opraci o jakiś zakodzony algorytm
   - W sumie jakoś to trzeba będzie zgrać, żeby te ficzery co wystawiamy dla SI nie wadziły w graniu przez człowieka
6. Plany na później
   - Jak coś mi przyjdzie do głowy to dopiszę
   - Zobaczymy co powie prowadzący na to co zrobimy
   - Ewentualnie można poczytać później o wizualizacji informacji z sieci neuronowej, pewnie takie grafiki fajnie by wyglądały w sprawodzaniu