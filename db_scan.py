import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

class BOT():
    def __init__(self):

        self.stop_words = set(stopwords.words("russian"))
        self.stemmer = SnowballStemmer("russian")
        self.user_input = None
        # Пример базы запросов пользователей
        self.user_queries = [
            "привет",
            "что ты умеешь?",
            "найди рецепт пиццы",
            "когда ближайшее мероприятие?",
            "есть новости?",
            "мероприятие",
            # и т.д.
        ]

        self.create_clusters()

    # Предобработка текстовых запросов (удаление стоп-слов и стемминг)
    def preprocess_text(self,text):
        words = nltk.word_tokenize(text.lower())
        words = [self.stemmer.stem(word) for word in words if word.isalpha() and word not in self.stop_words]
        return " ".join(words)

    def create_clusters(self):
        # Подготовка данных
        self.processed_data = [self.preprocess_text(text) for text in self.user_queries]

        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(self.processed_data)

        # Кластеризация с помощью K-Means на N кластеров
        self.kmeans = KMeans(n_clusters=5,n_init = "auto")
        clusters = self.kmeans.fit_predict(X)

        # Отображение результатов кластеризации
        for i, cluster_id in enumerate(clusters):
            print(f"    Запрос: {self.user_queries[i]}, Кластер: {cluster_id}")

    def main(self):
        while self.user_input != "Стоп":
            self.user_input = input("Введите что-то: ")
            processed_new_query = self.preprocess_text(self.user_input)
            new_query_vector = self.vectorizer.transform([processed_new_query])
            new_cluster = self.kmeans.predict(new_query_vector)[0]
            print(f"    Запрос: {self.user_input}, Кластер: {new_cluster}")

if __name__ == "__main__":
    work = BOT()
    work.main()