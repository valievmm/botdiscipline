"""
Анализ данных Telegram-бота «Трекер привычек» (numpy + pandas + matplotlib).
Запуск:  python analysis.py
Графики сохраняются в папку charts/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("charts")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.figsize": (8, 5), "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
BLUE, ORANGE, GRAY, GREEN, RED = "#2f6fb0", "#e08a2e", "#8a8f98", "#3a9d6b", "#c0443c"
pd.set_option("display.width", 150)
WD_ORDER = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


def pct(x):
    return f"{x * 100:.1f}%"


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ============ 1. ЗАГРУЗКА И ПЕРВИЧНЫЙ ОБЗОР ============
section("1. ПЕРВИЧНЫЙ ОБЗОР")
df = pd.read_csv("habits_data.csv", parse_dates=["date"])
print("shape:", df.shape)
print("\ndtypes:\n", df.dtypes)
print("\nПропуски:\n", df.isna().sum())
print("\nОписательные статистики (числовые):\n", df.describe().round(2))
print("\nКатегории:\n", df.category.value_counts())
print("\nПериод данных:", df.date.min().date(), "-", df.date.max().date())
print("Пользователей:", df.user_id.nunique(), "| привычек:", df.habit_id.nunique())

# ---- производные признаки ----
DATA_END = df.date.max()
df["signup"] = df.groupby("user_id").date.transform("min")          # дата регистрации = первая запись
df["week"] = (df.date - df.signup).dt.days // 7                      # неделя жизни пользователя
df["has_reminder"] = df.reminder_time.notna()
hour = pd.to_datetime(df.reminder_time, format="%H:%M").dt.hour
df["slot"] = np.select([hour.between(6, 10), hour.between(11, 16), hour.between(17, 20), hour >= 21],
                       ["утро (6–10)", "день (11–16)", "вечер (17–20)", "поздно (21+)"],
                       default="без напоминания")

users = df.groupby("user_id").agg(signup=("signup", "first"))
users["weeks_obs"] = ((DATA_END - users.signup).dt.days + 1) // 7    # сколько недель наблюдались полностью
users["cohort"] = users.signup.dt.month.map({7: "Июль", 8: "Август", 9: "Сентябрь"})
df = df.merge(users[["weeks_obs"]], left_on="user_id", right_index=True)

# «Активная неделя» = в эту неделю пользователь отметил хотя бы одну привычку
act = df.groupby(["user_id", "week"]).completed.sum().unstack(fill_value=0) > 0
act = act.reindex(index=users.index, columns=range(int(df.week.max()) + 1), fill_value=False)
act.columns.name = "week"
act_flag = act.stack().rename("active_week").reset_index()
df = df.merge(act_flag, on=["user_id", "week"], how="left")

# ============ 2. КЛЮЧЕВЫЕ МЕТРИКИ ============
section("2. КЛЮЧЕВЫЕ МЕТРИКИ")
overall = df.completed.mean()
active_rate = df.loc[df.active_week, "completed"].mean()
print("Общий % выполнения (все записи):", pct(overall))
print("% выполнения в активные недели (пока пользователь «в боте»):", pct(active_rate))
print("Доля записей в неактивные недели (тихий отток):", pct(1 - df.active_week.mean()))

# --- по пользователям: разная дисциплина ---
u_rate = df.groupby("user_id").completed.mean()
print("\nРаспределение % выполнения по пользователям:\n", u_rate.describe().round(3))
print("Пользователей с выполнением >=70%:", pct((u_rate >= .7).mean()),
      "| 40-70%:", pct(((u_rate >= .4) & (u_rate < .7)).mean()), "| <40%:", pct((u_rate < .4).mean()))

# --- серии ---
d2 = df.sort_values(["habit_id", "date"]).copy()
hab = d2.groupby("habit_id").agg(max_streak=("current_streak", "max"), rate=("completed", "mean"),
                                 n=("completed", "size"))
print("\nСредний current_streak по всем записям:", round(df.current_streak.mean(), 2))
print("Средний current_streak в записях с выполнением:", round(df.loc[df.completed == 1, "current_streak"].mean(), 2))
print("Максимальная серия по привычке: mean=%.1f median=%.0f p90=%.0f max=%d" % (
    hab.max_streak.mean(), hab.max_streak.median(), np.percentile(hab.max_streak, 90), hab.max_streak.max()))
for k in (3, 7, 14, 21, 30):
    print(f"  привычек с серией >= {k}: {pct((hab.max_streak >= k).mean())}")
# длины всех завершённых серий (run-length)
new_run = d2.completed.ne(d2.groupby("habit_id").completed.shift())
run_id = new_run.cumsum()
runs = d2[d2.completed == 1].groupby(run_id[d2.completed == 1]).size()
print("Всего серий:", len(runs), "| медианная длина:", runs.median(), "| доля серий длиной 1–2 дня:", pct((runs <= 2).mean()))

# --- retention ---
retention = pd.DataFrame({w: (act.loc[users.weeks_obs > w, w].mean(), (users.weeks_obs > w).sum())
                          for w in range(13)}, index=["retention", "n_users"]).T
print("\nRetention по неделям с регистрации (доля пользователей с >=1 выполнением в неделю):\n",
      retention.assign(retention=retention.retention.map(pct)))
cohort = pd.DataFrame(np.nan, index=["Июль", "Август", "Сентябрь"], columns=range(8))
for c in cohort.index:
    for w in cohort.columns:
        m = (users.cohort == c) & (users.weeks_obs > w)
        if m.sum() >= 5:
            cohort.loc[c, w] = act.loc[m, w].mean()
print("\nКогорты (месяц регистрации) x неделя:\n", (cohort * 100).round(0))
print("Размеры когорт:\n", users.cohort.value_counts())

# --- по категориям, дням, частоте ---
by_cat = df.groupby("category").completed.agg(["mean", "size"]).sort_values("mean", ascending=False)
print("\nПо категориям:\n", by_cat.assign(mean=by_cat["mean"].map(pct)))
by_wd = df.groupby("weekday").completed.agg(["mean", "size"]).reindex(WD_ORDER)
print("\nПо дням недели:\n", by_wd.assign(mean=by_wd["mean"].map(pct)))
print("\nПо целевой частоте:\n", df.groupby("target_frequency").completed.mean().map(pct))

# --- напоминания ---
rem = df[df.has_reminder].copy()
rem2 = rem.dropna(subset=["reminder_replied"]).copy()
rem2["replied"] = rem2.reminder_replied.astype(int)
reply_rate = rem2.replied.mean()
comp_rem = df.groupby("has_reminder").completed.mean()
comp_slot = df.groupby("slot").completed.mean()
reply_slot = rem2.groupby("slot").replied.mean()
by_reply = rem2.groupby("replied").completed.mean()
print("\nДоля привычек с напоминанием:", pct(df.drop_duplicates("habit_id").has_reminder.mean()))
print("Доля ответов на напоминания:", pct(reply_rate), "(записей с напоминанием:", len(rem), ", с пропуском ответа:",
      rem.reminder_replied.isna().sum(), ")")
print("Выполнение БЕЗ напоминания: ", pct(comp_rem[False]), "| С напоминанием:", pct(comp_rem[True]),
      "| разница, п.п.: %.1f" % ((comp_rem[True] - comp_rem[False]) * 100))
print("Выполнение, если был ответ=0 / 1:", by_reply.map(pct).to_dict())
print("Выполнение по времени напоминания:\n", comp_slot.map(pct))
print("Доля ответов по времени напоминания:\n", reply_slot.map(pct))

# --- нагрузка и серия ---
load = df.assign(n=df.active_habits_count.clip(upper=4)).groupby("n").completed.agg(["mean", "size"])
print("\nВыполнение по числу активных привычек (4 = 4+):\n", load.assign(mean=load["mean"].map(pct)))
d2["prev_streak"] = d2.groupby("habit_id").current_streak.shift(1).fillna(0)
bucket = pd.cut(d2.prev_streak, [-1, 0, 2, 6, 13, 10_000], labels=["0", "1–2", "3–6", "7–13", "14+"])
streak_eff = d2[d2.active_week].groupby(bucket, observed=True).completed.agg(["mean", "size"])
print("\nP(выполнить | серия накануне), только активные недели:\n", streak_eff.assign(mean=streak_eff["mean"].map(pct)))

# ============ 3. ГРАФИКИ ============
# 1) Выполнение по категориям
fig, ax = plt.subplots()
bars = ax.bar(by_cat.index, by_cat["mean"] * 100, color=BLUE, label="Выполнение в категории")
ax.axhline(overall * 100, color=ORANGE, ls="--", label=f"Среднее по всем: {overall * 100:.1f}%")
for b, n in zip(bars, by_cat["size"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.0f}%\n(n={n})", ha="center", fontsize=9)
ax.set_title("Процент выполнения по категориям привычек")
ax.set_xlabel("Категория"); ax.set_ylabel("Выполнено, %"); ax.set_ylim(0, 75); ax.legend()
plt.xticks(rotation=20); fig.tight_layout(); fig.savefig(OUT / "1_completion_by_category.png", dpi=150); plt.close(fig)

# 2) Динамика по неделям жизни + retention
dw = df[df.weeks_obs > df.week].groupby("week").completed.mean()
rt = retention[retention.n_users >= 15]
dw = dw[dw.index.isin(rt.index)]
fig, ax = plt.subplots()
ax.plot(rt.index, rt.retention * 100, marker="o", color=GREEN, label="Retention (≥1 выполнение за неделю)")
ax.plot(dw.index, dw.values * 100, marker="s", color=BLUE, label="Доля выполненных отметок")
ax.set_title("Динамика по неделям с момента регистрации")
ax.set_xlabel("Неделя после регистрации (0 = первая)"); ax.set_ylabel("%"); ax.set_ylim(0, 105); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "2_weekly_dynamics_retention.png", dpi=150); plt.close(fig)

# 3) Распределение длины серий
fig, ax = plt.subplots()
ax.hist(hab.max_streak, bins=range(0, int(hab.max_streak.max()) + 3, 2), color=BLUE, edgecolor="white", label="Привычки")
ax.axvline(hab.max_streak.mean(), color=ORANGE, ls="--", label=f"Среднее: {hab.max_streak.mean():.1f}")
ax.axvline(hab.max_streak.median(), color=RED, ls=":", label=f"Медиана: {hab.max_streak.median():.0f}")
ax.set_title("Распределение максимальной серии (streak) по привычкам")
ax.set_xlabel("Максимальная серия, дней подряд"); ax.set_ylabel("Число привычек"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "3_streak_distribution.png", dpi=150); plt.close(fig)

# 4) Дни недели
fig, ax = plt.subplots()
colors = [BLUE] * 5 + [ORANGE] * 2
ax.bar(by_wd.index, by_wd["mean"] * 100, color=colors)
ax.axhline(overall * 100, color=GRAY, ls="--")
for i, v in enumerate(by_wd["mean"] * 100):
    ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="Будни"), Patch(color=ORANGE, label="Выходные"),
                   plt.Line2D([], [], color=GRAY, ls="--", label="Среднее по всем дням")])
ax.set_title("Выполнение по дням недели"); ax.set_xlabel("День недели"); ax.set_ylabel("Выполнено, %")
ax.set_ylim(0, 70); fig.tight_layout(); fig.savefig(OUT / "4_completion_by_weekday.png", dpi=150); plt.close(fig)

# 5) Напоминания и выполнение
order = ["без напоминания", "утро (6–10)", "день (11–16)", "вечер (17–20)", "поздно (21+)"]
x = np.arange(len(order)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w / 2, comp_slot.reindex(order) * 100, w, color=BLUE, label="Доля выполненных отметок")
ax.bar(x[1:] + w / 2, reply_slot.reindex(order[1:]) * 100, w, color=ORANGE, label="Доля ответов на напоминание")
for i, v in enumerate(comp_slot.reindex(order) * 100):
    ax.text(i - w / 2, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9)
for i, v in enumerate(reply_slot.reindex(order[1:]) * 100, start=1):
    ax.text(i + w / 2, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(order, rotation=12)
ax.set_title("Напоминания: время отправки, ответы и выполнение")
ax.set_xlabel("Время отправки напоминания"); ax.set_ylabel("%"); ax.set_ylim(0, 80); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "5_reminders_vs_completion.png", dpi=150); plt.close(fig)

# 6) Когорты retention (heatmap)
fig, ax = plt.subplots(figsize=(8, 3.8))
cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("#eeeeee")
im = ax.imshow(np.ma.masked_invalid(cohort.values * 100), cmap=cmap, vmin=0, vmax=100, aspect="auto")
for i in range(cohort.shape[0]):
    for j in range(cohort.shape[1]):
        v = cohort.iloc[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", color="white" if v > 0.6 else "black")
ax.set_xticks(range(cohort.shape[1])); ax.set_yticks(range(cohort.shape[0])); ax.set_yticklabels(
    [f"{c} (n={users.cohort.value_counts()[c]})" for c in cohort.index])
ax.set_title("Retention по когортам (месяц регистрации), %")
ax.set_xlabel("Неделя после регистрации"); ax.set_ylabel("Когорта"); ax.grid(False)
fig.colorbar(im, ax=ax, label="% активных пользователей")
fig.tight_layout(); fig.savefig(OUT / "6_cohort_retention.png", dpi=150); plt.close(fig)

# 7) Нагрузка (число привычек) и серия накануне
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
axes[0].bar(["1", "2", "3", "4+"], load["mean"] * 100, color=BLUE, label="Выполнение")
axes[0].axhline(overall * 100, color=ORANGE, ls="--", label="Среднее по всем")
for i, v in enumerate(load["mean"] * 100):
    axes[0].text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9)
axes[0].set_title("Выполнение vs число активных привычек")
axes[0].set_xlabel("Активных привычек у пользователя"); axes[0].set_ylabel("Выполнено, %"); axes[0].set_ylim(0, 75); axes[0].legend()
axes[1].bar(streak_eff.index.astype(str), streak_eff["mean"] * 100, color=GREEN, label="Вероятность выполнить сегодня")
for i, v in enumerate(streak_eff["mean"] * 100):
    axes[1].text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9)
axes[1].set_title("Выполнение vs серия накануне (активные недели)")
axes[1].set_xlabel("Серия накануне, дней"); axes[1].set_ylabel("Выполнено, %"); axes[1].set_ylim(0, 100); axes[1].legend()
fig.tight_layout(); fig.savefig(OUT / "7_load_and_streak_effect.png", dpi=150); plt.close(fig)

print("\nГрафики сохранены в", OUT.resolve())
