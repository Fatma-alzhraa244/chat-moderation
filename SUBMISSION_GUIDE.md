# دليل التسليم – SWE015 (Use Case 4: Real-Time Chat Moderation)

> الموعد النهائي: الأحد 22 نوفمبر 2026. التقييم 0 / 50 / 100 فقط، فأي حاجة ناقصة بتنزل الدرجة.

## قاعدة ذهبية
كل رقم في الـ slides لازم يطلع من `results.csv` بتاع **تشغيلك انت**. لو كتبت رقم من عندك أو من AI يبقى "fabricated results" = صفر.

---

## الخطوات بالترتيب

### 1) GitHub (10 دقايق)
1. اعمل حساب على github.com لو معندكش.
2. New repository → الاسم `chat-moderation` → **Public** → Create.
3. "uploading an existing file" → اسحب الملفات: `common.py`, `train.py`, `app.py`, `requirements.txt`, `README.md`, `.gitignore`, `colab_run.ipynb` → Commit.
4. خد لينك الريبو (مثلا `https://github.com/اسمك/chat-moderation`).

### 2) Colab – التدريب (20 دقيقة)
1. افتح colab.research.google.com → File → Upload notebook → `colab_run.ipynb`.
2. Runtime → Change runtime type → **T4 GPU**.
3. شغّل الخلايا بالترتيب (Shift+Enter). غيّر `GITHUB_URL` للينك الريبو بتاعك.
4. خلية `python train.py` هي الأطول. في الآخر هتطبع جدول النتائج و هل الـ latency budget اتحقق ولا لأ.
5. لو الـ INT8 معداش الـ budget (50 ms): غيّر الرقم بصدق (مثلا `--budget_ms 100`) واشرح في الـ slides. الدكتور قال الـ budget انت اللي بتحدده، بس لازم تكتبه صراحة وتقول اتحقق ولا لأ.

### 3) Hugging Face (10 دقايق)
1. حساب على huggingface.co → Settings → Access Tokens → New token → النوع **Write**.
2. شغّل خلايا 4 (login) و 5 (upload) في الـ notebook. غيّر `HF_REPO` لاسمك.
3. ارجع للـ README في GitHub وبدّل `<your-hf-username>/<your-artifacts-repo>` بالاسم الحقيقي.
4. حمّل ملفات النتايج (خلية 5) وارفعها على GitHub: `results.csv`, `results.md`, `hardware.txt`, `data_stats.json`, `failures_int8.csv`.

### 4) اختبر الـ README (مهم – شرط Gate 2)
افتح Colab جديد فاضي، ونفّذ حرفيا أوامر الـ README (clone → pip install → set ARTIFACTS_REPO → python app.py). لو اشتغل يبقى تمام. لو لأ، صلّح الـ README.

### 5) الفيديو (دقيقتين بالظبط)
سجّل بـ OBS أو Zoom (Share screen + Record) أو Win+G. **لازم صوتك يشرح.**
- 0:00–0:15: "الموقع بتاع ألعاب محتاج يكشف الرسايل المسيئة في أقل من 50ms على CPU."
- 0:15–1:15: التطبيق شغال لايف. اكتب رسالتين على الأقل، منهم واحدة صعبة (مثلا leetspeak زي `y0u 4re st00pid` أو حروف متكررة). غيّر الموديل من القائمة وورّيه الـ latency log.
- 1:15–1:45: ورّي الـ architecture diagram وجدول النتائج.
- 1:45–2:00: الخلاصة (مين كسب، مين خسر، والـ quantization عمل إيه في الحجم والسرعة).
ارفعه على YouTube **Unlisted** أو Drive (Anyone with the link). افتح اللينك في **Private/Incognito window** وتأكد إنه بيفتح.

### 6) الـ Slides (PDF، من 8 لـ 12 سلايد)
| # | المحتوى |
|---|---|
| 1 | اسمك، الـ student ID، "Use Case 4 – Real-Time Chat Moderation"، لينك الريبو، لينك الفيديو |
| 2 | **Introduction:** المشكلة (منصة ألعاب)، لمين، يعني إيه نجاح (macro-F1 + latency < budget) |
| 3 | **Development:** الداتا (TweetEval offensive – الحجم والـ splits من `data_stats.json`) والـ preprocessing |
| 4 | Architecture diagram (من الـ mermaid في README) – لازم يطابق الكود |
| 5 | **Technique map:** جدول يوضح كل component بيستخدم أنهي technique وفي أنهي area (A/B/C/D) |
| 6 | الموديلات الأربعة + الـ rule baseline وإزاي اتدرّبوا (epochs, lr, SGD…) |
| 7 | طريقة الـ quantization (dynamic INT8 على الـ Linear layers) |
| 8 | **Results table** (من `results.md`: macro-F1، size MB، latency ms، والـ hardware) |
| 9 | **Conclusion:** كل موديل بيكسب فين وبيخسر فين |
| 10 | Limitations + **مثال فشل واحد على الأقل** (من `failures_int8.csv`) + إيه الخطوة الجاية |
| 11 | **AI Usage** (مثال تحت) |
| 12 | References (TweetEval, OLID, DistilBERT, HF libraries) |

**شريحة AI Usage (عدّلها بصدق):**
"Claude (Anthropic) was used to draft the Python code (preprocessing, training script, Gradio app), the README and the slide outline. I ran all training and measurements myself, and every number in this presentation comes from my own runs. I reviewed and can explain every component."

---

## تشيك ليست الدكتور (لو واحدة ناقصة = 50)
- [ ] التطبيق بياخد input جديد وقت العرض (R1)
- [ ] 4 areas كلها موجودة + سلايد technique map (R2)
- [ ] 4 موديلات على نفس الـ test set (R3)
- [ ] جدول: metric + size + latency (≥100 run) + hardware (R4)
- [ ] التطبيق بيشغّل الـ quantized model (R5) – الافتراضي في `app.py`
- [ ] الـ architecture diagram مطابق للكود (R6)
- [ ] README شغال في بيئة نضيفة + `requirements.txt` + مفيش weights في الريبو (R7)
- [ ] سلايد AI usage (R8)
- [ ] Introduction / Development / Conclusion + اللينكين في السلايد الأول
- [ ] الفيديو ≤ 2:00، بصوت، فيه التطبيق والـ architecture والنتايج
- [ ] لينك الفيديو والريبو بيفتحوا في Private window

## سؤال للـ AI زي ما الدكتور قال
الصق في أي AI: "ده نص الواجب [الصق الملف]. وده الـ repo/slides/النتايج بتوعي [الصق]. راجع كل شرط في Gate 1 و Gate 2 وقولي لو فيه حاجة ناقصة."

## لو حصلت مشكلة
ابعتلي نص الإيرور بالكامل (copy/paste) وأنا أصلّحه. وبعد ما تشغّل `train.py` ابعتلي `results.md` وأنا أجهزلك الـ slides بالأرقام الحقيقية.
