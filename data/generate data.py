"""
Генератор синтетических данных для бота «Трекер привычек».

Идея: сначала моделируем ПОЛЬЗОВАТЕЛЕЙ (дисциплина, уход, «провалы»),
потом их ПРИВЫЧКИ, потом каждый ДЕНЬ каждой привычки (отметка, напоминание, серия).
Все допущения вынесены в константы вверху — их можно менять и объяснять на защите.

Запуск:  python generate_data.py   ->  habits_data.csv
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------- Параметры симуляции ----------------
N_USERS = 72                       # зарегистрировано; в CSV попадают только те, кто создал привычку (получается 71)
SIGNUP_START = date(2026, 7, 1)    # первые регистрации
SIGNUP_END = date(2026, 9, 15)     # последние регистрации
DATA_END = date(2026, 10, 31)      # конец наблюдения
GRACE_DAYS = 7                     # сколько дней бот «стучится» после ухода пользователя
WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

# Каталог привычек: (название, категория, частота)
DAILY, MULTI = "ежедневно", "несколько раз в неделю"
CATALOG = [
    ("Выпить 2 л воды", "здоровье", DAILY), ("Принять витамины", "здоровье", DAILY),
    ("Прогулка 30 минут", "здоровье", DAILY), ("Без сладкого", "здоровье", DAILY),
    ("Зарядка 15 минут", "спорт", DAILY), ("Растяжка 10 минут", "спорт", DAILY),
    ("Тренировка в зале", "спорт", MULTI), ("Пробежка 5 км", "спорт", MULTI),
    ("Читать 20 страниц", "учёба", DAILY), ("Английский 20 минут", "учёба", DAILY),
    ("Задача по Python", "учёба", MULTI), ("Конспект лекции", "учёба", MULTI),
    ("План на день", "продуктивность", DAILY), ("Без соцсетей до 12:00", "продуктивность", DAILY),
    ("Глубокая работа 1 час", "продуктивность", MULTI), ("Разобрать почту", "продуктивность", MULTI),
    ("Записать расходы", "финансы", DAILY), ("Отложить 100 ₽", "финансы", MULTI),
    ("Лечь до 23:30", "сон", DAILY), ("Подъём в 7:00", "сон", DAILY),
    ("Без телефона за час до сна", "сон", DAILY),
    ("Медитация 10 минут", "другое", DAILY), ("Дневник благодарности", "другое", DAILY),
    ("Позвонить родителям", "другое", MULTI),
]
# Вероятности выбора категории при создании привычки
CAT_WEIGHTS = {"здоровье": 0.18, "спорт": 0.20, "учёба": 0.22, "продуктивность": 0.16,
               "финансы": 0.07, "сон": 0.10, "другое": 0.07}
# Сложность категории (добавка к вероятности выполнения)
CAT_MOD = {"здоровье": 0.05, "спорт": -0.06, "учёба": -0.02, "продуктивность": -0.04,
           "финансы": -0.08, "сон": -0.08, "другое": 0.02}
# Эффект дня недели (Пн..Вс) по категориям: выходные тяжелее для учёбы/работы
WD_MOD = {
    "учёба": [0.02, 0.02, 0.01, 0.0, -0.03, -0.12, -0.10],
    "продуктивность": [0.03, 0.03, 0.01, 0.0, -0.04, -0.14, -0.12],
    "здоровье": [0, 0, 0, 0, -0.01, -0.05, -0.05],
    "спорт": [0, 0, 0, 0, -0.02, 0.0, 0.02],
    "финансы": [0, 0, 0, 0, 0, -0.04, -0.03],
    "сон": [0, 0, 0, 0, -0.05, -0.12, -0.04],
    "другое": [0, 0, 0, 0, 0, -0.02, 0.01],
}
# Время напоминания и его эффект на выполнение
REMINDER_TIMES = ["07:30", "08:00", "09:00", "12:30", "14:00", "18:00", "20:00", "21:30", "22:30"]
REMINDER_BOOST = {"07:30": 0.10, "08:00": 0.10, "09:00": 0.09, "12:30": 0.07, "14:00": 0.06,
                  "18:00": 0.08, "20:00": 0.07, "21:30": 0.04, "22:30": 0.02}
P_HAS_REMINDER = 0.75              # доля привычек с включёнными напоминаниями
P_DELIVERY_FAIL = 0.015            # ошибки доставки -> пропуск в reminder_replied


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def make_users():
    """Создаём пользователей: дата регистрации, дисциплина, момент ухода, период «провала»."""
    days_range = (SIGNUP_END - SIGNUP_START).days + 1
    # Больше регистраций в начале сентября (начало учебного года)
    w = np.ones(days_range)
    for i in range(days_range):
        d = SIGNUP_START + timedelta(days=i)
        if date(2026, 9, 1) <= d <= date(2026, 9, 10):
            w[i] = 3.0
    w /= w.sum()
    signup_offsets = rng.choice(days_range, size=N_USERS, p=w)

    users = []
    for i in range(N_USERS):
        arche = rng.choice(["высокая", "средняя", "низкая"], p=[0.25, 0.45, 0.30])
        disc = {"высокая": rng.normal(0.80, 0.05),
                "средняя": rng.normal(0.62, 0.07),
                "низкая": rng.normal(0.42, 0.08)}[arche]
        weekly_h = {"высокая": 0.03, "средняя": 0.07, "низкая": 0.14}[arche]
        # Недельная вероятность ухода: ранний отток выше всего
        churn_day = None
        for wk in range(0, 20):
            h = weekly_h + (0.08 if wk == 0 else 0.10 if wk == 1 else 0.0)
            if rng.random() < h:
                churn_day = wk * 7 + int(rng.integers(0, 7))
                break
        # Временный «провал» (экзамены, болезнь, отпуск) у 30% пользователей
        slump = None
        if rng.random() < 0.30:
            start = int(rng.integers(10, 50))
            slump = (start, start + int(rng.integers(7, 15)))
        users.append(dict(user_id=f"U{i+1:03d}",
                          signup=SIGNUP_START + timedelta(days=int(signup_offsets[i])),
                          disc=clip(disc, 0.2, 0.92), churn_day=churn_day, slump=slump))
    return users


def make_habits(user, counter):
    """Привычки пользователя: 1-3 при регистрации, возможно ещё одна позже."""
    habits = []
    n_initial = int(rng.choice([1, 2, 3], p=[0.35, 0.40, 0.25]))
    offsets = [0] * n_initial
    if rng.random() < 0.45:                       # добавил привычку позже
        offsets.append(int(rng.integers(7, 40)))
    if rng.random() < 0.15:                       # и ещё одну
        offsets.append(int(rng.integers(20, 60)))
    used = set()
    cats = list(CAT_WEIGHTS)
    probs = np.array(list(CAT_WEIGHTS.values()))
    for off in offsets:
        for _ in range(20):
            cat = rng.choice(cats, p=probs / probs.sum())
            name, c, freq = CATALOG[int(rng.integers(len(CATALOG)))]
            if c == cat and name not in used:
                used.add(name)
                break
        else:
            continue
        counter[0] += 1
        has_rem = rng.random() < P_HAS_REMINDER
        sched = None
        if freq == MULTI:                         # 3-4 запланированных дня в неделю
            sched = set(rng.choice(7, size=int(rng.choice([3, 4])), replace=False).tolist())
        habits.append(dict(habit_id=f"H{counter[0]:04d}", name=name, category=cat, freq=freq,
                           created=user["signup"] + timedelta(days=off), sched=sched,
                           reminder=str(rng.choice(REMINDER_TIMES)) if has_rem else None))
    return habits


def simulate():
    rows = []
    counter = [0]
    for u in make_users():
        habits = make_habits(u, counter)
        streak = {h["habit_id"]: 0 for h in habits}
        day_shock = {}                            # общий «плохой/хороший день» пользователя
        d = u["signup"]
        last_active = {h["habit_id"]: None for h in habits}
        while d <= DATA_END:
            rel = (d - u["signup"]).days
            churned = u["churn_day"] is not None and rel >= u["churn_day"]
            stopped = u["churn_day"] is not None and rel >= u["churn_day"] + GRACE_DAYS
            if stopped:
                break
            day_shock[d] = rng.normal(0, 0.10)
            active = [h for h in habits if h["created"] <= d]
            n_active = len(active)
            for h in active:
                wd = d.weekday()
                if h["sched"] is not None and wd not in h["sched"]:
                    continue                      # у привычки «несколько раз в неделю» этот день не запланирован
                age = (d - h["created"]).days
                prev_streak = streak[h["habit_id"]]
                if churned:
                    done = 0
                else:
                    decay = 0.82 + 0.23 * np.exp(-age / 18)          # «медовый месяц» и спад
                    base = (u["disc"] + CAT_MOD[h["category"]] + (0.04 if h["freq"] == MULTI else 0)) * decay
                    p = base + WD_MOD[h["category"]][wd]
                    p += 0.005 * min(prev_streak, 14)                # серия мотивирует не рвать её
                    p -= 0.05 * max(0, n_active - 2)                 # перегруз привычками
                    if h["reminder"]:
                        p += REMINDER_BOOST[h["reminder"]]
                    p += day_shock[d]
                    if u["slump"] and u["slump"][0] <= rel <= u["slump"][1]:
                        p *= 0.45
                    done = int(rng.random() < clip(p, 0.02, 0.97))
                # Реакция на напоминание (нажатие любой кнопки: «Выполнено» / «Позже» / «Пропустить»)
                reminder_time, replied = h["reminder"], np.nan
                if reminder_time:
                    if churned:
                        replied = 0
                    elif done:
                        replied = int(rng.random() < clip(0.55 + 0.40 * u["disc"], 0.5, 0.95))
                    else:
                        replied = int(rng.random() < clip(0.10 + 0.15 * u["disc"], 0.05, 0.35))
                    if rng.random() < P_DELIVERY_FAIL:
                        replied = np.nan                              # доставка не подтверждена
                streak[h["habit_id"]] = prev_streak + 1 if done else 0
                rows.append(dict(user_id=u["user_id"], date=d.isoformat(), habit_id=h["habit_id"],
                                 habit_name=h["name"], category=h["category"],
                                 target_frequency=h["freq"], completed=done,
                                 reminder_time=reminder_time, reminder_replied=replied,
                                 current_streak=streak[h["habit_id"]], weekday=WEEKDAYS[wd],
                                 active_habits_count=n_active))
            d += timedelta(days=1)
    df = pd.DataFrame(rows)
    df["reminder_replied"] = df["reminder_replied"].astype("Int64")   # целые + пропуски
    return df.sort_values(["date", "user_id", "habit_id"]).reset_index(drop=True)


if __name__ == "__main__":
    df = simulate()
    df.to_csv("habits_data.csv", index=False, encoding="utf-8")
    print("Строк:", len(df), "| пользователей:", df.user_id.nunique(), "| привычек:", df.habit_id.nunique())
    print(df.head(10).to_string())
