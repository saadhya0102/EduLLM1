from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
import zipfile

import pymupdf

from src.data_prep_brazil import (
    _balanced_sample,
    _parse_block,
    parse_exam_pdf,
    parse_key_pdf,
    read_enade_responses,
    read_enem_responses,
)
from src.schema import Item


class Brazil8kPrepTests(unittest.TestCase):
    def test_option_block(self):
        parsed = _parse_block([
            "This is a sufficiently long assessment question.",
            "A) Alpha", "B) Beta", "C) Gamma", "D) Delta", "E) Epsilon",
        ])
        self.assertEqual(parsed, (
            "This is a sufficiently long assessment question.",
            ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        ))

    def test_pdf_question_and_key_parsers(self):
        with tempfile.TemporaryDirectory() as td:
            exam = Path(td) / "exam.pdf"
            key = Path(td) / "key.pdf"
            doc = pymupdf.open()
            page = doc.new_page()
            page.insert_textbox((50, 50, 550, 780), (
                "QUESTAO 1\nThis is a sufficiently long first question.\n"
                "A) One\nB) Two\nC) Three\nD) Four\n"
                "E) Five VALIDINEP2023VALIDINEP2023 footer\n"
                "QUESTAO 2\nThis is a sufficiently long second question.\n"
                "A) Red\nB) Blue\nC) Green\nD) Black\nE) White\n"
                "QUESTAO 1\nHow much time did you spend on this examination?\n"
                "A) One hour\nB) Two hours\nC) Three hours\n"
                "D) Four hours\nE) Five hours"
            ), fontsize=10)
            doc.save(exam)
            doc.close()
            doc = pymupdf.open()
            page = doc.new_page()
            page.insert_text(
                (50, 50), "QUESTAO 1 D\nQUESTAO 2 ***", fontsize=10)
            doc.save(key)
            doc.close()

            parsed = parse_exam_pdf(exam)
            self.assertEqual(sorted(parsed), [1, 2])
            self.assertIn("first question", parsed[1][0])
            self.assertEqual(parsed[1][1][-1], "Five")
            self.assertEqual(parse_key_pdf(key), {1: "D", 2: "9"})

    def test_response_aggregation_uses_exact_choices(self):
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "mock.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(
                    "DADOS/mock_arq1.txt",
                    "NU_ANO;CO_CURSO;CO_GRUPO\n2023;10;17\n")
                z.writestr(
                    "DADOS/mock_arq3.txt",
                    "CO_CURSO;DS_VT_GAB_OFG_FIN;DS_VT_GAB_OCE_FIN;"
                    "DS_VT_ESC_OFG;DS_VT_ESC_OCE\n"
                    "10;ABCDE;ABCDE;ABCDE;BBCDE\n"
                    "10;ABCDE;ABCDE;BBCDE;ABCDE\n")
            data = read_enade_responses(archive)
            self.assertEqual(data.groups["17"].key(), "ABCDE")
            self.assertEqual(
                data.groups["17"].position_counts[0], {"A": 1, "B": 1})

    def test_quoted_lgpd_headers_are_normalized(self):
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "lgpd.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(
                    "microdados_2019_LGPD/2.DADOS/microdados2019_arq1.txt",
                    '"NU_ANO";"CO_CURSO";"CO_GRUPO"\n"2019";"10";"17"\n')
                z.writestr(
                    "microdados_2019_LGPD/2.DADOS/microdados2019_arq3.txt",
                    '"CO_CURSO";"DS_VT_GAB_OFG_FIN";'
                    '"DS_VT_GAB_OCE_FIN";"DS_VT_ESC_OFG";'
                    '"DS_VT_ESC_OCE"\n'
                    '"10";"ABCDE";"ABCDE";"ABCDE";"BBCDE"\n')
            data = read_enade_responses(archive)
            self.assertEqual(data.groups["17"].key(), "ABCDE")
            self.assertEqual(
                data.groups["17"].position_counts[0], {"B": 1})

    def test_response_placeholders_do_not_shift_positions(self):
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "fixed-width.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(
                    "DADOS/mock_arq1.txt",
                    "CO_CURSO;CO_GRUPO\n10;17\n")
                z.writestr(
                    "DADOS/mock_arq3.txt",
                    "CO_CURSO;DS_VT_GAB_OFG_FIN;DS_VT_GAB_OCE_FIN;"
                    "DS_VT_ESC_OFG;DS_VT_ESC_OCE\n"
                    "10;ABCDE;A8CDE;ABCDE;A*BDE\n"
                    "10;ABCDE;A8CDE;ABCDE;A.CDE\n")
            data = read_enade_responses(archive)
            group = data.groups["17"]
            self.assertEqual(group.key(), "A8CDE")
            self.assertEqual(group.position_counts[1], {})
            self.assertEqual(group.position_counts[2], {"B": 1, "C": 1})

    def test_balanced_sample_size(self):
        rows = []
        for group in ("a", "b"):
            for i in range(5):
                rows.append(Item(
                    id=f"{group}{i}", question="Question long enough",
                    options=["1", "2", "3", "4", "5"],
                    dist=[0.2] * 5, correct_index=0,
                    source="ENADE 2019", skill=group))
        picked = _balanced_sample(rows, 6, seed=13)
        self.assertEqual(len(picked), 6)
        self.assertEqual({x.skill for x in picked}, {"a", "b"})

    def test_enem_form_position_aggregation(self):
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "enem.zip"
            item_rows = [
                "CO_POSICAO;SG_AREA;CO_ITEM;CO_PROVA;TX_GABARITO",
            ]
            for pos in range(1, 46):
                item_rows.append(f"{pos};CN;{1000+pos};77;A")
            responses = "A" * 45
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(
                    "DADOS/ITENS_PROVA_2017.csv",
                    "\n".join(item_rows) + "\n")
                z.writestr(
                    "DADOS/MICRODADOS_ENEM_2017.csv",
                    "CO_PROVA_CN;TX_RESPOSTAS_CN;TP_PRESENCA_CN\n"
                    f"77;{responses};1\n77;{'B' + responses[1:]};1\n")
            data = read_enem_responses(archive)
            self.assertEqual(data.item_at[("CN", "77", 1)], "1001")
            self.assertEqual(data.counts["1001"], {"A": 1, "B": 1})


if __name__ == "__main__":
    unittest.main()
