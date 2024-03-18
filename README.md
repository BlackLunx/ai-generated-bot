### 1. Создание конфигурационного файла
 - Создайте файл .env в корне директории и поместите туда следующую строчку:
 `BOT_TOKEN = <YOR TG BOT TOKEN>`

### 2. Создание модели
Существует два варианта:
 - Скачайте [обученную модель](https://drive.google.com/drive/folders/1TYZJt0apGgne1VPqIy4qOy3q8f01vxZ-?usp=sharing) и поместите его в папку `deberta` в корне проекта
 - Запустите файл `src/model/train.py` чтобы обучить модель.
### 3. Запуск бота
 - Установите необходимые библиотеки из `requirements.txt`
 - Запустите бота 
 ```Bash
 export PYTHONPATH=$PYTHONPATH:$PWD/src/
 python src/bot/bot.py
 ```
