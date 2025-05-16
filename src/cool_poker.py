import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import itertools
import random
from collections import Counter

class WinProba():

    def __init__(self):

        self.COMBINATION_NAMES = {
            10: "Royal flush",
            9: "Straight flush",
            8: "Four of a kind",
            7: "Full house",
            6: "Flush",
            5: "Straight",
            4: "Three of a kind",
            3: "Two pair",
            2: "Pair",
            1: "High card"
        }

        # Определение колоды
        self.SUITS = ['h', 'd', 'c', 's']  # червы, бубны, трефы, пики
        self.RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.RANK_VALUES = {r: i + 1 for i, r in enumerate(self.RANKS)}
    
    def evaluate_hand(self, cards):
        """
        Оценивает комбинацию из 5-7 карт и возвращает оценку руки.
        Чем выше число, тем сильнее комбинация.
        """
        # Получаем все возможные комбинации из 5 карт
        all_five_cards_combinations = list(itertools.combinations(cards, 5))
        
        # Находим лучшую комбинацию из 5 карт
        best_score = 0
        
        for five_cards in all_five_cards_combinations:
            score = self.evaluate_five_cards(five_cards)
            if score > best_score:
                best_score = score
        return best_score
    
    def evaluate_five_cards(self, cards):
        """Оценивает комбинацию из 5 карт по системе, где более сильная рука имеет большее значение."""
        if len(cards) != 5:
            ValueError("There must be exactly 5 cards for evaluation")
        
        # Извлекаем ранги и масти
        ranks = [card[0] for card in cards]
        suits = [card[1] for card in cards]
        
        # Преобразуем ранги в числовые значения
        rank_values = [self.RANK_VALUES[r] for r in ranks]
        
        # Проверяем флеш
        is_flush = len(set(suits)) == 1
        
        # Проверяем стрит
        sorted_values = sorted(rank_values)
        
        # Проверка на стрит A-5
        if set(sorted_values) == {2, 3, 4, 5, 14}:
            is_straight = True
            straight_high = 5  # Высшая карта в стрите A-5 - это 5
        else:
            is_straight = (len(set(sorted_values)) == 5 and 
                          max(sorted_values) - min(sorted_values) == 4)
            straight_high = max(sorted_values) if is_straight else 0
        
        # Подсчет повторяющихся рангов
        rank_count = Counter(rank_values)
        
        # Сортировка карт по количеству и затем по значению
        cards_by_count = sorted(rank_count.items(), key=lambda x: (x[1], x[0]), reverse=True)
        
        # Определение комбинации
        combination_level = 0
        secondary_values = 0
        
        # Роял-флеш
        if is_straight and is_flush and straight_high == 14:
            combination_level = 10
        
        # Стрит-флеш
        elif is_straight and is_flush:
            combination_level = 9
            secondary_values = straight_high
        
        # Каре
        elif cards_by_count[0][1] == 4:
            combination_level = 8
            four_of_kind = cards_by_count[0][0]
            kicker = cards_by_count[1][0]
            secondary_values = four_of_kind * 10**6 + kicker
        
        # Фулл-хаус
        elif cards_by_count[0][1] == 3 and cards_by_count[1][1] == 2:
            combination_level = 7
            three_of_kind = cards_by_count[0][0]
            pair = cards_by_count[1][0]
            secondary_values = three_of_kind * 10**6 + pair * 10**4
        
        # Флеш
        elif is_flush:
            combination_level = 6
            # Сортируем значения карт для флеша от высшей к низшей
            flush_values = sorted(rank_values, reverse=True)
            secondary_values = flush_values[0] * 10**8 + flush_values[1] * 10**6 + \
                               flush_values[2] * 10**4 + flush_values[3] * 10**2 + flush_values[4]
        
        # Стрит
        elif is_straight:
            combination_level = 5
            secondary_values = straight_high
        
        # Тройка
        elif cards_by_count[0][1] == 3:
            combination_level = 4
            three_of_kind = cards_by_count[0][0]
            kicker1 = cards_by_count[1][0]
            kicker2 = cards_by_count[2][0]
            secondary_values = three_of_kind * 10**6 + kicker1 * 10**4 + kicker2 * 10**2
        
        # Две пары
        elif cards_by_count[0][1] == 2 and cards_by_count[1][1] == 2:
            combination_level = 3
            high_pair = max(cards_by_count[0][0], cards_by_count[1][0])
            low_pair = min(cards_by_count[0][0], cards_by_count[1][0])
            kicker = cards_by_count[2][0]
            secondary_values = high_pair * 10**6 + low_pair * 10**4 + kicker * 10**2
        
        # Пара
        elif cards_by_count[0][1] == 2:
            combination_level = 2
            pair = cards_by_count[0][0]
            kickers = sorted([card[0] for card in cards_by_count[1:]], reverse=True)
            secondary_values = pair * 10**6 + kickers[0] * 10**4 + kickers[1] * 10**2 + kickers[2]
        
        # Старшая карта
        else:
            combination_level = 1
            high_cards = sorted(rank_values, reverse=True)
            secondary_values = high_cards[0] * 10**8 + high_cards[1] * 10**6 + \
                               high_cards[2] * 10**4 + high_cards[3] * 10**2 + high_cards[4]
        
        # Финальная оценка с увеличенным множителем для combination_level
        return combination_level * 10**10 + secondary_values
    
    def validate_cards(self, all_cards, context=""):
        """
        Проверяет корректность карт и отсутствие дубликатов.
        
        Parameters:
        - all_cards: список карт для проверки
        - context: контекст проверки для более информативных сообщений об ошибках
        
        Returns:
        - list: список валидных карт
        
        Raises:
        - ValueError: если найдены некорректные карты или дубликаты
        """
        
        def parse_card(card):
            """Parses a card into rank and suit."""
            if len(card) != 2:
                raise ValueError(f"Invalid card format: {card}. Use 'rank+suit' format, for example 'Ah'")
            rank, suit = card[0], card[1]
            if rank not in self.RANKS:
                raise ValueError(f"Invalid rank: {rank}. Valid ranks: {', '.join(self.RANKS)}")
            if suit not in self.SUITS:
                raise ValueError(f"Invalid suit: {suit}. Valid suits: {', '.join(self.SUITS)}")
            return rank, suit
        
        # Удаляем пустые строки
        valid_cards = [card for card in all_cards if card]
        
        # Проверка формата карт
        for card in valid_cards:
            parse_card(card)
        
        # Проверка на дубликаты
        if len(valid_cards) != len(set(valid_cards)):
            # Находим дубликаты
            seen = set()
            duplicates = [card for card in valid_cards if card in seen or seen.add(card)]
            raise ValueError(f"{context}Duplicate cards detected: {', '.join(duplicates)}")
        
        # Проверка количества карт в зависимости от контекста
        if context == "Player: ":
            if not (1 <= len(valid_cards) <= 2):
                raise ValueError(f"{context}You must have 1 or 2 cards in hand")
        elif context == "Board: ":
            if not (0 <= len(valid_cards) <= 5):
                raise ValueError(f"{context}There must be 0 to 5 cards on the board")
        elif "Opponent" in context:
            if not (1 <= len(valid_cards) <= 2):
                raise ValueError(f"{context}Opponent must have 1 or 2 cards in hand")
        
        return valid_cards

    def prepare_simulation(self, my_cards, table_cards, num_opponents, opponents_cards=None):
        """
        Подготавливает данные для симуляции.
        
        Parameters:
        - my_cards: список карт игрока
        - table_cards: список карт на столе
        - num_opponents: количество оппонентов
        - opponents_cards: список списков карт оппонентов (опционально)
        
        Returns:
        - tuple: (валидные карты игрока, валидные карты стола (всегда 5 карт), 
                  списки карт оппонентов, оставшаяся колода)
        """
        
        def create_deck():
            """Создает полную колоду карт."""
            return [r + s for r in self.RANKS for s in self.SUITS]
        
        def validate_all_cards(my_cards, table_cards, opponents_cards, num_opponents):
            """Проверяет все карты и возвращает их валидированные версии."""
            if num_opponents < 1:
                raise ValueError("Должен быть хотя бы один оппонент")
            
            # Валидируем карты игрока и стола
            valid_my_cards = self.validate_cards(my_cards, "Player: ")
            valid_table_cards = self.validate_cards(table_cards, "Board: ")
            
            # Валидируем карты оппонентов, если они предоставлены
            valid_opponents_cards = []
            if opponents_cards:
                for i, opponent_cards in enumerate(opponents_cards):
                    valid_opponent_cards = self.validate_cards(opponent_cards, f"Opponent {i+1}: ")
                    valid_opponents_cards.append(valid_opponent_cards)
                
                # Проверяем, что число оппонентов соответствует предоставленным картам
                if len(valid_opponents_cards) > num_opponents:
                    raise ValueError(f"The number of provided opponent hands ({len(valid_opponents_cards)}) exceeds the specified number of opponents ({num_opponents})")
            
            return valid_my_cards, valid_table_cards, valid_opponents_cards
        
        def check_duplicates(all_card_sets):
            """Проверяет наличие дубликатов среди всех карт."""
            all_cards = []
            for card_set in all_card_sets:
                all_cards.extend(card_set)
                
            if len(all_cards) != len(set(all_cards)):
                seen = set()
                duplicates = [card for card in all_cards if card in seen or seen.add(card)]
                raise ValueError(f"Duplicate cards detected between different players or the board: {', '.join(duplicates)}")
        
        def complete_table_cards(valid_table_cards, deck):
            """Добавляет недостающие карты на стол."""
            full_table_cards = valid_table_cards.copy()
            cards_needed_on_table = 5 - len(valid_table_cards)
            
            if cards_needed_on_table > 0:
                # Перемешиваем колоду для выбора карт на стол
                random.shuffle(deck)
                table_cards_to_add = deck[:cards_needed_on_table]
                full_table_cards.extend(table_cards_to_add)
                # Удаляем взятые карты из колоды
                deck = [card for card in deck if card not in table_cards_to_add]
            
            return full_table_cards, deck
        
        def complete_opponents_hands(valid_opponents_cards, num_opponents, deck):
            """Заполняет руки оппонентов недостающими картами."""
            all_opponents_cards = valid_opponents_cards.copy()
            remaining_opponents = num_opponents - len(all_opponents_cards)
            
            if remaining_opponents > 0:
                for _ in range(remaining_opponents):
                    # Берем 2 карты из колоды для оппонента
                    opponent_cards = [deck.pop(0), deck.pop(0)]
                    all_opponents_cards.append(opponent_cards)
            
            return all_opponents_cards, deck
        
        # Выполняем основные этапы подготовки симуляции
        valid_my_cards, valid_table_cards, valid_opponents_cards = validate_all_cards(
            my_cards, table_cards, opponents_cards, num_opponents
        )
        
        # Проверяем дубликаты
        check_duplicates([valid_my_cards, valid_table_cards] + valid_opponents_cards)
        
        # Создаем колоду и удаляем известные карты
        deck = create_deck()
        all_known_cards = valid_my_cards + valid_table_cards
        for opponent_cards in valid_opponents_cards:
            all_known_cards.extend(opponent_cards)
        
        for card in all_known_cards:
            if card in deck:
                deck.remove(card)
        
        # Добавляем недостающие карты на стол
        full_table_cards, deck = complete_table_cards(valid_table_cards, deck)
        
        # Заполняем руки оппонентов
        all_opponents_cards, deck = complete_opponents_hands(valid_opponents_cards, num_opponents, deck)
        
        return valid_my_cards, full_table_cards, all_opponents_cards, deck

    def run_simulation(self, my_cards, table_cards, opponents_cards):
        """
        Проводит одну симуляцию раздачи на основе уже подготовленных карт.
        
        Parameters:
        - my_cards: список карт игрока
        - table_cards: список 5 карт на столе
        - opponents_cards: список списков карт оппонентов
        
        Returns:
        - tuple: (результат, моя комбинация, комбинации оппонентов, индекс лучшего оппонента)
        """
        
        def get_combination_type(score):
            """Извлекает тип комбинации из оценки руки"""
            # Тип комбинации - это первая цифра в оценке, разделенной на 10^10
            return score // (10**10)
        
        # Оцениваем руку игрока
        my_hand = my_cards + table_cards
        my_score = self.evaluate_hand(my_hand)
        my_combination = get_combination_type(my_score)
        
        # Оцениваем руки оппонентов
        opponents_scores = []
        opponents_combinations = []
        
        for opponent_cards in opponents_cards:
            opponent_hand = opponent_cards + table_cards
            opponent_score = self.evaluate_hand(opponent_hand)
            opponent_combination = get_combination_type(opponent_score)
            
            opponents_scores.append(opponent_score)
            opponents_combinations.append(opponent_combination)
        
        # Определяем результат
        max_opponent_score = max(opponents_scores) if opponents_scores else 0
        max_opponent_index = opponents_scores.index(max_opponent_score) if opponents_scores else -1
        if my_score > max_opponent_score:
            result = "win"
        elif my_score == max_opponent_score:
            result = "tie"
        else:
            result = "loss"
        
        return result, my_combination, opponents_combinations, max_opponent_index
    
    def calculate_win_probability(self, my_cards, table_cards, num_opponents, opponents_cards=None, num_simulations=50):
            """
            Рассчитывает вероятность победы в покере и собирает статистику по комбинациям.
            
            Parameters:
            - my_cards: список из 2 строк, представляющих ваши карты
            - table_cards: список из 0-5 строк, представляющих карты на столе
            - num_opponents: количество оппонентов
            - opponents_cards: список списков карт оппонентов (опционально)
            - num_simulations: количество симуляций для оценки вероятности
            
            Returns:
            - tuple: (вероятность_победы, вероятность_ничьей, вероятность_поражения, статистика_комбинаций)
            """
            # Счетчики для статистики
            win_count = 0
            tie_count = 0
            loss_count = 0
            
            # Словари для отслеживания комбинаций
            winning_combinations = Counter()  # Изменено: отслеживаем комбинации, которые принесли победу
            lost_to_combinations = Counter()
            my_combinations = Counter()
            
            # Запускаем симуляции
            for _ in tqdm(range(num_simulations)):
                # Подготавливаем данные для симуляции (для каждой симуляции новый расклад)
                valid_my_cards, full_table_cards, all_opponents_cards, deck = self.prepare_simulation(
                    my_cards, table_cards, num_opponents, opponents_cards)
                
                # Запускаем симуляцию
                result, my_combination, opponents_combinations, max_opponent_index = self.run_simulation(
                    valid_my_cards, full_table_cards, all_opponents_cards)
                # Обновляем статистику
                my_combinations[my_combination] += 1
                if result == "win":
                    win_count += 1
                    # Записываем нашу комбинацию, которая принесла победу
                    winning_combinations[my_combination] += 1
                elif result == "tie":
                    tie_count += 1
                else:  # loss
                    loss_count += 1
                    # Записываем комбинацию, которой мы проиграли
                    lost_to_combinations[opponents_combinations[max_opponent_index]] += 1
            # Форматируем и возвращаем статистику
            return self.format_statistics(
                win_count, tie_count, loss_count, 
                my_combinations, winning_combinations, lost_to_combinations)


    def format_statistics(self, win_count, tie_count, loss_count, my_combinations, defeated_combinations, lost_to_combinations):
        """
        Форматирует статистику для вывода.
        
        Parameters:
        - win_count, tie_count, loss_count: счетчики результатов
        - my_combinations: словарь с комбинациями игрока
        - defeated_combinations: словарь с побежденными комбинациями
        - lost_to_combinations: словарь с комбинациями, которым проиграл
        
        Returns:
        - tuple: (вероятность_победы, вероятность_ничьей, вероятность_поражения, статистика_комбинаций)
        """
        
        # Рассчитываем вероятности
        total = win_count + tie_count + loss_count
        win_probability = win_count / total if total > 0 else 0
        tie_probability = tie_count / total if total > 0 else 0
        loss_probability = loss_count / total if total > 0 else 0
        
        # Преобразуем числовые значения комбинаций в их названия
        defeated_combinations_named = {self.COMBINATION_NAMES[k]: v for k, v in defeated_combinations.items()}
        lost_to_combinations_named = {self.COMBINATION_NAMES[k]: v for k, v in lost_to_combinations.items()}
        my_combinations_named = {self.COMBINATION_NAMES[k]: v for k, v in my_combinations.items()}
        
        # Создаем статистику комбинаций
        combinations_stats = {
            "my_combinations": my_combinations_named,
            "defeated_combinations": defeated_combinations_named,
            "lost_to_combinations": lost_to_combinations_named
        }
        
        return win_probability, tie_probability, loss_probability, combinations_stats
    
    def plot_combinations_statistics(self, combo_stats, show=True):
        """
        Строит столбчатые диаграммы для статистики покерных комбинаций.
        
        Parameters:
        - combo_stats: словарь с данными о комбинациях
        """
        
        # Определяем порядок комбинаций от сильной к слабой
        combinations_order = [self.COMBINATION_NAMES[i] for i in range(1, 11)]
        
        # Извлекаем данные из combo_stats
        my_combinations = {combo: 0 for combo in combinations_order}
        defeated_combinations = {combo: 0 for combo in combinations_order}
        lost_to_combinations = {combo: 0 for combo in combinations_order}
        
        # Заполняем данными из статистик
        for combo, count in combo_stats["my_combinations"].items():
            if combo in my_combinations:
                my_combinations[combo] = count
        
        for combo, count in combo_stats["defeated_combinations"].items():
            if combo in defeated_combinations:
                defeated_combinations[combo] = count
        
        for combo, count in combo_stats["lost_to_combinations"].items():
            if combo in lost_to_combinations:
                lost_to_combinations[combo] = count
        
        # Создаем график
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Poker Combinations Statistics', fontsize=16)
        
        # Индексы для позиционирования столбцов
        x = np.arange(len(combinations_order))
        width = 0.8  # Ширина столбцов
        
        # Функция для добавления значений над столбцами
        def add_values_on_bars(ax, counts):
            for i, v in enumerate(counts):
                ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        
        # График моих комбинаций
        counts_my = [my_combinations[combo] for combo in combinations_order]
        bars1 = ax1.bar(x, counts_my, width, color='skyblue')
        ax1.set_title('My All Combinations')
        ax1.set_xticks(x)
        ax1.set_xticklabels(combinations_order, rotation=45, ha='right')
        ax1.set_ylabel('Count')
        add_values_on_bars(ax1, counts_my)
        
        # График побежденных комбинаций
        counts_defeated = [defeated_combinations[combo] for combo in combinations_order]
        bars2 = ax2.bar(x, counts_defeated, width, color='green')
        ax2.set_title('My Winning Combinations')
        ax2.set_xticks(x)
        ax2.set_xticklabels(combinations_order, rotation=45, ha='right')
        ax2.set_ylabel('Count')
        add_values_on_bars(ax2, counts_defeated)
        
        # График проигрышных комбинаций
        counts_lost = [lost_to_combinations[combo] for combo in combinations_order]
        bars3 = ax3.bar(x, counts_lost, width, color='salmon')
        ax3.set_title('Combinations That Defeated Me')
        ax3.set_xticks(x)
        ax3.set_xticklabels(combinations_order, rotation=45, ha='right')
        ax3.set_ylabel('Count')
        add_values_on_bars(ax3, counts_lost)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if show:
            plt.show()
