Факультатив C++, проектная работа \
Белов Тимофей, Б13-303, МФТИ, ВШПИ

Неделя 17 (07.02.2025-13.03.2025) \
Пообновлял код в визуализации, чтобы поддерживать немного больше функционала под тесты, которые планируем сделать. \
Только mean_divergence штука оказалась не тем, чем задумывалось. Она просто показывает, насколько неустоичивы предсказания детектора. Там берется среднее в секции и потом среднее отклонение от этого среднего. Полезно немного, но задумывалось, что будет что-то другое, но я уже не могу ни вспомнить, ни понять что (потому что увидел это). \
Сделал код для тестирования сразу же партии экспериментов. Обнаружил несколько незначительных багов в utils и vtk наших. \
Очень много страдал над этапами калибрации, которые я сам там сделал. Оказалось что настройка внутренных параметров камеры в vtk какая-то кривая (мой способ, который нашел в интернете). Я сначала понял, что очень неправильные значения по z получаются в результате калибровки позиции камеры. Я попробовал подставить идеальные, которые идут на настройку vtk и оказалось, что есть устойчивое отклонение в пару сантиметров. После того, как исправил первый этап калибровки камеры (оказалось что нужны очень близкие расстояния при калибровке на шахматке), начал получать постоянно чуть смещенные значения от моих идеальных. Но на этих смещенных следующие этапы работали с меньше ошибкой. \
Только очень много неочевидных тестов и запусков приходилось делать, чтобы это выявить. \
В этой папке еще можно увидеть картинки, которые сейчас можно получить на основе этих экспериментов.