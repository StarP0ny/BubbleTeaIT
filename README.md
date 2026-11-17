# BubbleTeaIT

Локальный координатор контекста для AI-агентов: управляет готовыми движками
поиска и навигации по проектам через единый CLI `bti` и MCP. Возвращает
проверяемые фрагменты с проектом, путями, строками и SHA.
Режим неизменяемых снимков точных версий также сохраняется.

Проект не зависит от VS Code и конкретного харнеса. Индекс версии модуля
переиспользуется несколькими составами; внутренние базы движков не сливаются.

## Версия 0.1.0

Для рабочего поиска по нескольким checkout доступен
[режим координатора готовых движков](docs/engines.md): один CLI/MCP, общий
CocoIndex Code runtime, отдельные индексы проектов и постоянные read-only
сессии Serena с выбираемым clangd/ccls. Машинный пример:
[config-service-engines.example.toml](configs/config-service-engines.example.toml).
Прежний режим точных снимков сохраняется.

Рабочий baseline:

- регистрация источников, immutable-снимки и составы точных версий;
- семантический поиск Qwen3-Embedding 0.6B и старый адаптер CocoIndex/E5;
- прямой поиск и чтение с цитируемыми evidence-ID;
- навигация по C/C++ через standalone Serena;
- локальный сервис с watch, автосинхронизацией и workspace-поиском;
- stdio MCP.

Направление исследователя вынесено из baseline в локальный `attic/`:
его код, эксперименты и тесты не входят в Git. Результаты проверки поиска по документам и границы оценки приведены ниже.

## Быстрый старт

Нужен Python 3.12+ и Git.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[mcp]'
.\.venv\Scripts\python.exe -m bubbleteait init
.\.venv\Scripts\python.exe -m bubbleteait doctor
```

В проекте создаётся `bti.toml` (см. [конфиги проектов](docs/project-config.md)),
затем:

```powershell
bti plan
bti sync
bti project-info
```

Для постоянного наблюдения и MCP:

```powershell
bti watch -p C:/work/controller
bti --project-dir C:/work/controller mcp
```

Установка `bti` в PATH и настройка моделей — в [инструкции запуска](docs/setup.md).

## MCP

В режиме координатора доступны девять инструментов. `context_search` возвращает
фрагмент с версией и готовыми аргументами `next_read` для чтения всей
распознанной функции или соседних строк. Для навигации есть `project_info`,
`list_files`, `code_symbols`, для чтения — `read_sources` и `resolve_sources`,
для наблюдения и синхронизации — `watch_status`, `sync_status`, `sync_project`.
Подключение Codex к выбранному workspace описано в [настройке MCP](docs/setup.md#mcp).

В режиме снимков (`mcp --snapshots`) доступны 13 инструментов:

- прямой контекст: `context_search`, `read_sources`, `resolve_sources`;
- навигация: `list_sources`, `list_files`, `code_symbols`;
- точный доступ: `search`, `read_source`, `index_status`, `project_info`;
- сервис: `watch_status`, `sync_status`, `sync_project`.

Рабочий маршрут агента: `context_search` → при необходимости `read_sources`.
Каталог и `resolve_sources` нужны только для навигации и метаданных.

Сервис, lease, workspace и `freshness=wait` описаны в
[документе сервиса](docs/service.md).

## Данные и границы

- Runtime-данные по умолчанию лежат вне checkout:
  `%LOCALAPPDATA%/BubbleTeaIT` (Windows) или `$XDG_DATA_HOME/bubbleteait`.
- `.local/` — локальные снимки, прогоны и receipts; в Git не входит.
- `docs/research/`, `docs/evals/`, `evals/`, `profiles/`, `infra/` —
  локальные исследовательские материалы; в baseline не входят.
- `attic/` — вынесенные из baseline исторические документы, тесты и
  benchmark-скрипты. Файлы остаются на диске, но не коммитятся.
- Веса моделей, индексы и секреты в Git не хранятся.

## Компоненты

Исходники сторонних компонентов закреплены Git-сабмодулями:

```powershell
git submodule update --init --depth 1
```

Список и ревизии — в [components.md](docs/components.md). Наличие сабмодуля
не означает, что сервис установлен или запущен; установка выполняется по
инструкциям upstream.

## Проверки

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Baseline-набор не требует облачной модели, GPU или сети. Часть тестов
использует фикстуры реальных снимков; отдельные проверки могут пропускаться
при отсутствии необязательных зависимостей — skips нужно учитывать отдельно.

Проверка реального stdio-клиента:

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/probe_mcp.py --composition <composition-id> --query "heating control" --channel text --output .local/mcp-check.json
```

## Ограничения

- В сравнении на 24 вопросах по 51 Markdown-разделу CocoIndex нашёл ожидаемый раздел
  в первой пятёрке для 23 вопросов, гибридный QMD — для 21. Вопросы и целевые разделы
  зафиксированы до прогона. Проверка сквозных инженерных задач на независимой выборке
  остаётся в [плане работ](TODO.md).
- Проверка композиции всё ещё читает байты снимков на каждом запросе; на
  крупном корпусе это основной вклад в latency. Кэш и stat-fast-path — в
  [TODO](TODO.md).
- Сервис использует периодическую сверку корпуса, а не файловые уведомления
  ОС; стоимость сверки больших SDK не измерена.
- Исследователь не входит в baseline; его материалы остаются локальными.

## Документы

- [Иерархические конфиги](docs/project-config.md).
- [Сервис и watch](docs/service.md).
- [Установка и модели](docs/setup.md).
- [Компоненты и ревизии](docs/components.md).
- [План работ](TODO.md).
- [Правила работы с репозиторием](AGENTS.md).
