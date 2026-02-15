# Модуль 16. Використання баз даних
# Тема: Використання баз даних. Частина 5
# ===================================================================================================
# Завдання 1
# Створіть додаток «Соціальна мережа», який зберігає інформацію про користувача, його друзів, публікації користувача.
# Зберігайте дані у базі даних NoSQL. Можете використовувати Redis в якості платформи.
# ===================================================================================================
import redis
import json


class SocialApp:
    def __init__(self):
        self.server = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.current_user = None

    def _get_password_key(self, username):
        return f'password:{username}'

    def _get_friend_key(self, username):
        return f'friends:{username}'

    def _get_data_key(self, username):
        return f"data:{username}"

    # ■ додати користувача;
    def add_user(self, username, password):
        password_key = self._get_password_key(username)

        if self.server.exists(password_key):
            print("This user is already exists")
            return

        self.server.set(f'password:{username}', password)

    # ■ вхід за логіном і паролем;
    def login(self, username, password):
        password_key = self._get_password_key(username)

        if not self.server.exists(password_key):
            print("This user doesn't exist")
            return

        true_password = self.server.get(password_key)

        if password == true_password:
            print("Logged to the system")
            self.current_user = username
        else:
            print("Wrong password")

    # ■ редагувати інформацію про користувача;
    def add_user_data(self, user_data):
        if self.current_user is None:
            print("Log to the system")
            return

        data_key = self._get_data_key(self.current_user)

        for key, value in user_data.items():
            value = json.dumps(value)
            self.server.hset(data_key, key, value)

        print(f"Data for user {self.current_user} has been added")

    # ■ додати друзів користувачу;
    def add_friend(self, friend_name):
        if self.current_user is None:
            print("Log to the system")
            return

        friend_key = self._get_password_key(friend_name)
        if not self.server.exists(friend_key):
            print(f"This user with name {friend_name} doesn't exist")

        key1 = self._get_friend_key(friend_name)
        self.server.sadd(key1, self.current_user)

        key2 = self._get_friend_key(self.current_user)
        self.server.sadd(key2, friend_name)

    # ■ видалити користувача;
    def delete_user(self):
        if self.current_user is None:
            print("Log to the system")
            return

        key = self._get_password_key(self.current_user)
        self.server.delete(key)

        data_key = self._get_data_key(self.current_user)
        self.server.delete(data_key)

        friends_key = self._get_friend_key(self.current_user)
        friends = self.server.smembers(friends_key)

        for friend in friends:
            friend_key = self._get_friend_key(friend)
            self.server.srem(friend_key, self.current_user)

        self.server.delete(friends_key)
        self.current_user = None

        print("Account has been deleted")

    # ■ пошук користувача за ПІБ;
    def search_user(self, full_name):
        key_pattern = self._get_data_key('*')
        data_keys = self.server.keys(key_pattern)

        for data_key in data_keys:
            user_data = self.server.hgetall(data_key)
            print(user_data)

            if "name" not in user_data:
                continue

            user_name = user_data['name']

            user_name = json.loads(user_name)

            if user_name == full_name:
                print("This user exists in the system")
                print("User data")

                data = json.dumps(user_data, indent=4)

                print(data)
                return

        print("This user doesn't exist")

    # ■ перегляд інформації про користувача;

    # ■ перегляд усіх друзів користувача;

    # ■ перегляд усіх публікацій користувача.



# Testing ===============================================================
user_data_alex = {
            'name': 'Alex Shaposhnykov',
            'age': 23,
            'articles': ['article1', 'article2', 'article3']
        }

user_data_john= {
            'name': 'John Smith',
            'age': 27,
            'articles': ['article1', 'article2', 'article3']
        }

user_data_maria= {
            'name': 'Maria Sun',
            'age': 20,
            'articles': ['article1', 'article2', 'article3']
        }

app = SocialApp()

app.add_user('Alex', '111222')
app.login('Alex', '111222')

app.add_user("John", '666')
app.add_user("Maria", '222')

app.add_friend('Maria')
app.add_friend('John')

app.delete_user()

app.login('John', '666')
app.add_user_data(user_data_john)

app.login('Maria', '222')
app.add_user_data(user_data_maria)

app.search_user('John Smith')

# ===================================================================================================
# Завдання 2
# Створіть додаток «Музей літератури».
# Додаток має зберігати інформацію про експонати та людей, які мають відношення до експонатів.
# Можливості додатку:
# ■ вхід за логіном і паролем;
# ■ додати експонат;
# ■ видалити експонат;
# ■ редагування інформації про експонат;
# ■ перегляд повної інформації про експонат;
# ■ виведення інформації про всі експонати;
# ■ перегляд інформації про людей, які мають відношення до певного експонату;
# ■ перегляд інформації про експонати, що мають відношення до певної людини;
# ■ перегляд набору експонатів на основі певного критерію. Наприклад, показати всі книжкові експонати.
# Зберігайте дані у базі даних NoSQL. Можете використовувати Redis в якості платформи.
# ===================================================================================================
# Завдання 3 (за бажанням)
# Створіть додаток «Записна книжка», який зберігає інформацію про нотатки користувача. Можливості додатку:
# ■ вхід за логіном і паролем;
# ■ додавати нотатки;
# ■ видаляти нотатки;
# ■ редагувати нотатки;
# ■ переглядати нотатки. Якщо нотаток складається з декількох частин, відобразіть усі частини;
# ■ показ усіх нотаток;
# ■ перегляд нотаток за певний проміжок часу;
# ■ відображення нотаток, що містять набір заданих слів.
# Дані необхідно зберігати у базі даних NoSQL. Можете використовувати Redis в якості платформи.