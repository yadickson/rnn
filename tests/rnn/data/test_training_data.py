from unittest import TestCase

from faker import Faker

from rnn.data.training_data import TrainingData


class TestTrainingData(TestCase):

    def setUp(self) -> None:
        self.faker = Faker()
        self.epochs = self.faker.random.randint(10, 20)
        self.learning_rate = self.faker.random.randint(30, 40)
        self.generator = TrainingData(self.epochs, self.learning_rate)

    def test_should_check_epochs_are_assigned(self) -> None:
        self.assertEqual(self.epochs, self.generator.epochs)

    def test_should_check_learning_rate_is_assigned(self) -> None:
        self.assertEqual(self.learning_rate, self.generator.learning_rate)
