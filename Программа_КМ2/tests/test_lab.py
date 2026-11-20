import unittest

from pdn_course.recognizer import Recognizer
from pdn_course.validators import inn_valid, luhn_valid


class LabTests(unittest.TestCase):
    def test_required_attributes(self):
        text = (
            "Иванов Иван, ИНН 7707083893, карта 2202 2007 3549 3775, "
            "телефон +7 (916) 123-45-67, e-mail ivanov@example.ru."
        )
        labels = {entity.label for entity in Recognizer().recognize(text)}
        self.assertTrue({"SURNAME", "NAME", "INN", "BANK_CARD", "PHONE", "EMAIL"}.issubset(labels))

    def test_output_format(self):
        result = Recognizer().as_lists("Иван получил кредит")
        self.assertEqual(result, [["NAME", 0, 4]])

    def test_fio_is_recognized_by_model_without_name_lists(self):
        text = "Договор подписан Куклачёвым Юрием Дмитриевичем."
        entities = Recognizer().recognize(text)
        actual = [(item.label, item.text) for item in entities]
        self.assertEqual(
            actual,
            [
                ("SURNAME", "Куклачёвым"),
                ("NAME", "Юрием"),
                ("PATRONYMIC", "Дмитриевичем"),
            ],
        )

    def test_person_model_uses_context_for_homonym(self):
        entities = Recognizer().recognize("Роман написал роман.")
        self.assertEqual([(item.label, item.text) for item in entities], [("NAME", "Роман")])

    def test_patronymic_is_recognized_in_ambiguous_cases(self):
        examples = (
            ("Договор подписан Смирновым Александром Александровичем.", "Александровичем"),
            ("В заявлении о Новикове Петре Александровиче есть ошибка.", "Александровиче"),
        )
        for text, patronymic in examples:
            with self.subTest(text=text):
                entities = Recognizer().recognize(text)
                actual = [(item.label, item.text) for item in entities]
                self.assertIn(("PATRONYMIC", patronymic), actual)

    def test_checksums(self):
        self.assertTrue(inn_valid("7707083893"))
        self.assertFalse(inn_valid("7707083894"))
        self.assertTrue(luhn_valid("2202 2007 3549 3775"))
        self.assertFalse(luhn_valid("2202 2007 3549 3774"))

    def test_no_overlaps(self):
        entities = Recognizer().recognize("Иванов Иван: ivanov@example.ru")
        for left, right in zip(entities, entities[1:]):
            self.assertLessEqual(left.end, right.start)

    def test_bio_has_one_tag_per_token(self):
        tags = Recognizer().bio_tags("Иванов Иван указал ИНН 7707083893")
        self.assertTrue(all(isinstance(token, str) and isinstance(tag, str) for token, tag in tags))


if __name__ == "__main__":
    unittest.main()
