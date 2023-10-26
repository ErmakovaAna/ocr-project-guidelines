# Инструкция: Создание коллекции текстов в машиночитаемом формате с помощью технологии OCR

В инструкции дается пошаговое описание процесса создания корпуса текстов в машиночитаемом формате с помощью различных инструментов OCR. Полную версию можно найти [здесь](https://sysblok.ru/courses/kak-raspoznat-teksty-i-sdelat-korpus-dlja-issledovanija-poshagovaja-instrukcija/).

## Содержание
1. Сбор корпуса оцифрованных текстов
2. Распознавание текстов с помощью OCR
3. Оценка качества результатов OCR

## Дополнительные материалы к разделу 3. Оценка качества результатов OCR   

- `OCR_quality.ipynb`: ноутбук с примером реализации функции вычисления метрик **precision**,  **recall**, **F1-score**.
- `ocr_quality.py`: Python скрипт для выполнения вычислений через командную строку; пример команды запуска `python ocr_quality.py -o ocr_output.txt -gt ground_truth.txt`. Замените `ocr_output.txt` на путь к выходному файлу OCR и `ground_truth.txt` на путь к файлу с эталонным текстом.
  
