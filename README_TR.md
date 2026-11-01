# Split or Enrich: gradyana dayalı uzman sayısı seçicisinin önkayıtlı sentetik sınaması

Sabit bir parametre bütçesini "daha çok uzman" ile "daha büyük uzman" arasında nasıl bölmek gerektiğini, eğitimden önce yapılan ucuz bir ölçümün söyleyip söyleyemeyeceğini sınayan küçük ve önkayıtlı bir deney. Sonuç olumsuz.

[English version: README.md](README.md)

## Kısa sonuç

- Dondurulmuş deney (24 sentetik ortam, 432 eğitim) önceden belirlenmiş **NO_ADDED_UTILITY** kararıyla sonuçlandı.
- Önerilen seçici iyi yapılandırmalar seçti: 24 ortamın 24'ünde seçimi, eğitilmiş en iyi seçeneğe izin verilen tolerans içinde yakındı. Ancak basit bir rakip yöntem de aynı derecede iyiydi (belirleyici alt kümede ortalama regret'i hatta biraz daha düşüktü). Bu nedenle protokolün daha fazla yatırımdan önce şart koştuğu ek doğruluk faydası gösterilemedi.
- Gerçek bir dil modeli üzerinde planlanan devam aşaması ("Stage B") için karar **NO-GO** oldu. Stage B deneyi hiç çalıştırılmadı.
- Bu sonuç yöntemin hiçbir zaman işe yaramayacağını kanıtlamaz. Sınırlı bir sonuçtur: tek bir sentetik ve doğrusal problem ailesi, hafif girdi anizotropisi ve önceden belirlenmiş tek bir karar kuralı. Söylediği tek şey, yöntemin bu düzenekte daha fazla yatırımı hak etmediğidir.

## Problem, gündelik dille

Dört iş için yardımcı tutacağınızı ve bütçenizin sabit olduğunu düşünün. Dört işin hepsini yapan tek bir çok yetenekli kişi, ikişer iş üstlenen iki orta düzey kişi ya da her biri tek işe bakan dört yeni başlayan tutabilirsiniz. Toplam ücret her durumda aynı.

Hangisinin daha iyi olduğu işin niteliğine bağlıdır. Dört iş aslında aynı karmaşık işse, tek yetenekli kişi en iyisidir; bütçeyi bölmek her yeni başlayanı bu iş için fazla zayıf bırakır. Dört iş basit ama birbirinden farklıysa, dört yeni başlayan en iyisidir; tek kişi sürekli farklı yönlere çekilir. Soru şudur: Her seçeneği tam olarak denemeye para harcamadan, kısa bir ilk bakışla hangi durumda olduğunuzu anlayabilir misiniz?

Makine öğrenmesi diliyle "kişiler" bir mixture-of-experts katmanındaki uzmanlardır, "yetenek" her uzmana verilen parametre miktarıdır, "kısa ilk bakış" ise eğitim başlamadan birkaç örnek üzerinde hesaplanan kayıp gradyanıdır.

Burada kullanılan somut oyuncak sürüm: dört görev var; her biri 64 boyutlu girdiyi 64 boyutlu çıktıya götüren doğrusal bir dönüşüm. Her aday modelin tam olarak 1.024 eğitilebilir sayısı var. Adaylar: kapasitesi 8 olan tek uzman (E1/r8), her biri 4 kapasiteli iki uzman (E2/r4; dört görevi iki gruba ayırmanın yedi yolunun hepsi) ya da her biri 2 kapasiteli dört uzman (E4/r2). Toplam dokuz aday.

## Sınanan hipotez

Görev başına 64 "pilot" örnekten alınan ilk gradyanlarla hesaplanan bir skor ("naif Kronecker skoru"):

1. eğitilmiş test kaybı, eğitilmiş dokuz adayın en iyisine yakın olan bir yapılandırma seçecek,
2. girdiler izotropik olmadığında (farklı yönlerde farklı varyans taşıdığında) bunu güçlü ve basit alternatiflerden açıkça daha iyi yapacaktı.

Başarı bile tek başına bir Stage B'yi haklı çıkarmayacak, yalnız bir dil modeli aşamasının tasarlanmasını değerlendirmeye izin verecekti.

## Yöntem ne yapıyor?

**Sade anlatımla.** Her aday iki tür hata yapar. Farklı davranış gerektiren görevler aynı uzmanı paylaşırsa uzman ancak bir uzlaşma öğrenebilir ("tek kişiye fazla farklı iş"). Bir uzmanın kapasitesi yetersizse tek bir işi bile iyi temsil edemez ("iş, kişi için fazla karmaşık"). Skor bu iki hatayı her görevin ilk gradyanından tahmin eder ve tahmini toplamı en küçük olan adayı seçer.

**Kesin tanımıyla.** Her uzman, 8x8 matrislerin r adet Kronecker çarpımının toplamı olarak yazılan 64x64 bir matristir. Bir terim 128 parametre tutar; r terimli E uzman 128·E·r = 1.024 parametre eder. Satır-örnek düzeninde t görevinin negatif ilk gradyanı, pilot veriden P_t = Yᵀ X / n olarak bulunur. Görev grupları e (ağırlığı π_e, grup ortalaması P̄_e) olan bir aday için skor

S(c) = ½ Σ_e [ Σ_{t∈e} w_t ‖P_t − P̄_e‖²_F + π_e Σ_{j>r_e} s_j²(R(P̄_e)) ]

biçimindedir. Burada R, bir Kronecker çarpımını rank-1 matrise götüren yeniden düzenlemedir; s_j tekil değerlerdir. İlk terim grup içi uyuşmazlık, ikincisi Kronecker yaklaşımının kuyruk hatasıdır. İzotropik girdide ve kesin (popülasyon) gradyanlarla bu ifade, aday sınıfının tam minimum fazla riskine eşittir. Kesin tanımlar [PROTOCOL.md](PROTOCOL.md) dosyasındadır.

## Neyle karşılaştırıldı, karar kuralı neydi?

Bütün yöntemler aynı dokuz eğitilmiş aday arasından seçim yapar (her aday ayrı eğitim verisiyle iki kez, 400 tam-yığın Adam adımıyla eğitildi).

| Karşılaştırıcı | Rolü |
|---|---|
| Kovaryans düzeltmeli skor: gradyan, aynı pilotdan tahmin edilen (ridge düzenlileştirmeli) girdi kovaryansının tersiyle çarpıldıktan sonra aynı skor | Güçlü karşılaştırıcı, karara girer |
| Kısa eğitim: dokuz adayın hepsi 32 pilot örnekle kısa süre eğitilir, diğer 32 ile doğrulanır; ortam başına ortak 1,0 saniyelik üst sınır | Güçlü karşılaştırıcı, karara girer |
| Dokuz sabit adayın her biri (her zaman aynısını seçmek) | Karşılaştırıcı, karara girer; sonradan bakıldığında en iyisi E1/r8 |
| İki pilot yarısıyla crossfit skoru | İkincil, yalnız raporlanır |
| Yalnız-varyans ve yalnız-kuyruk ablasyonları, yarım-pilot güvenilirliği, oracle skor | Yalnız tanı amaçlı |

Regret, seçilen adayın test kaybı ile eğitilmiş en iyi adayın test kaybı arasındaki farkın Z'ye bölümüdür. Z, her zaman sıfır tahmin eden modelin gürültü dahil beklenen kaybıdır. Yani 0,01 regret, bu referans kaybın %1'i demektir.

Eşikler bilimsel veri oluşmadan önce sabitlendi (PROTOCOL.md bölüm 10–12):

1. Geçerlilik: bütün testler ve nümerik kontroller geçer; 24 ortamın hepsi 3.600 saniye içinde tamamlanır.
2. İyi seçim: 24 ortamın en az 20'sinde seçicinin regret'i en fazla 0,02'dir.
3. Ek fayda: 12 anizotropik ortamda seçicinin ortalama regret'i, en iyi karşılaştırıcının ortalamasından en az 0,01 düşük olmalıdır.

Üçüncü eşik geçilemezse karar NO_ADDED_UTILITY olur. Karar ayrıca beş duyarlılık görünümünde hesaplandı (iki başlangıcın ortalaması, başlangıç 0, başlangıç 1, aday başına minimum, kesin popülasyon riski). Görünümler farklı karar verseydi sonuç INCONCLUSIVE olacaktı.

24 ortam, 12 bağımsız "öğretmen" bloğundan oluşur; her blok bir kez izotropik, bir kez anizotropik girdiyle (koşul sayısı 4) çalıştırıldı. Bunlar 24 bağımsız tekrar değildir.

## Sonuçlar

Ana koşu: `runs/scientific_01/`. "Aniso ortalama", 12 anizotropik ortamdaki ortalama normalize regret'tir.

| Eşik | Koşul | Ölçüm | Sonuç | Kaynak (alan) |
|---|---|---|---|---|
| 1. Geçerlilik | testler, nümerik kontroller, 24 ortam, ≤ 3.600 s | geçerli, tam, 112,50 s, 59 test, 0 atlanan | geçti | `runs/scientific_01/summary.json` (`valid`, `complete`, `total_runtime_s`, `unit_tests`) |
| 2. İyi seçim | 24'te en az 20 ortamda regret ≤ 0,02 | 24/24; en kötü 0,0030454 | geçti | `summary.json` (`verdict.gates_by_aggregation.mean.gate2_envs_within_threshold`, `descriptive_test_regret.naive.all.max`) |
| 3. Ek fayda | naif ≤ en iyi karşılaştırıcı − 0,01 = −0,0096196 | naif 0,0003992; kovaryans 0,0003804 | kaldı | `summary.json` (`verdict.gates_by_aggregation.mean.primary_aniso_mean_regret`, `gate3_strongest_mean`) |
| Karar | | 5 görünümün hepsinde NO_ADDED_UTILITY | | `summary.json` (`verdict.verdict`, `verdict.labels_by_aggregation`) |

| Yöntem | Aniso ortalama regret | Medyan seçim süresi | Kaynak |
|---|---:|---:|---|
| Naif Kronecker skoru (sınanan yöntem) | 0,000399225 | 3,660 ms | `summary.json` `gate3_baseline_aniso_means` / `primary_aniso_mean_regret`; süreler: `runs/scientific_01/timing.csv` içindeki `naive_median_s`, `covariance_median_s`, `crossfit_median_s`, `early_ceiling_wall_s` sütunlarının 24 ortam üzerindeki medyanı |
| Kovaryans düzeltmeli skor | 0,000380419 | 4,591 ms | aynı |
| Kısa eğitim, 1 s üst sınır | 0,000957333 | 1.002,067 ms | aynı |
| Crossfit (ikincil) | 0,001593191 | 17,965 ms | `summary.json` `descriptive_test_regret.crossfit.aniso.mean` |
| En iyi sabit aday (E1/r8) | 0,021003398 | yok (sabit) | `summary.json` `gate3_baseline_aniso_means.fixed_0` |

Aynı koşudan diğer bulgular:

- Adaptif seçim, her zaman en iyi sabit yapılandırmayı kullanmaktan açıkça daha iyiydi: anizotropik ortalamada 0,021003 − 0,000399 ≈ 0,020604 Z (yukarıdaki iki `summary.json` değerinin farkı).
- Ortam başına en iyi aday (iki başlangıcın ortalaması): 12 ortamda E1, 12 ortamda bir E2 bölümlenmesi, hiçbir ortamda E4 (`summary.json` `winner_diversity.all.best_mean`; aday 0 E1, aday 1–7 E2, aday 8 E4). Yani "dört küçük uzman" rejimi hiçbir zaman kazanan olarak sınanmış olmadı.
- İki eğitim başlangıcının kazananı 24 ortamın 21'inde uyuştu (`winner_diversity.all.seed_winner_agreement`). İki pilot yarısının seçimleri 24'te 16 kez uyuştu; yarıdan yarıya ortalama Spearman sıra korelasyonu 0,654 (`summary.json` `half_pilot_reliability` içindeki 24 kayıttan hesaplandı).
- Kısa eğitim karşılaştırıcısına ortak 1 saniyelik bir üst sınır verildi; tüketilen hesap eşitlenmedi. Naif yöntemin süresiyle eşlenmiş sürüm, tek bir tur bile daha uzun sürdüğü için 24 ortamın hepsinde NOT_MEASURABLE çıktı (`timing.csv` `early_time_matched_status`).
- Bağımsız denetim: `analysis/independent_audit.json`, çalışma modüllerini içe aktarmadan yeniden hesaplayarak `status` PASS, 29 dondurulmuş kaynak hash'inin doğrulandığını, 432 satırı, en büyük yeniden hesaplanmış test kaybı farkının 7,0×10⁻¹⁶ Z olduğunu ve beş görünümün hepsinde aynı NO_ADDED_UTILITY kararını raporluyor.

Üçüncü eşiğin burada neden erişilemez olduğu: kovaryans karşılaştırıcısının regret'i (0,00038) zaten 0,01'in çok altında. Sıfır regret'li kusursuz bir seçici bile onu 0,01 farkla geçemezdi. Bu görüldükten sonra eşik değiştirilmedi. Naif ve kovaryans yöntemleri arasındaki çok küçük fark ne üstünlüğün ne de eşdeğerliğin kanıtıdır.

## Proje neden devam ettirilmedi?

- Belirleyici eşik geçilemedi: basit, kovaryansı dikkate alan bir sezgisel yönteme göre doğruluk faydası gösterilemedi.
- Dört küçük uzman rejimi hiç kazanmadı; dolayısıyla üç yönlü "böl ya da zenginleştir" anlatısı yalnız E1 ile E2 arasında sınanmış oldu.
- Temel bileşenler literatürde zaten var (Kronecker uzman karışımları, ilk gradyan hizalamasıyla Kronecker adapter bileşen tasarımı, uzmanlar arasında gradyana dayalı rank dağıtımı). Olumlu bir sonuç bile daha net bir özgünlük gerekçesine ihtiyaç duyacaktı.
- Ölçülen sonuçları değerlendiren üç yapay zekâ sohbet asistanının üçü de dil modeli aşaması için NO-GO önerdi, ancak gerekçeleri farklıydı. Koordinatörün kararı, pilotu dondurulmuş etiketiyle kapatmak ve Stage B'yi başlatmamaktı. Yeni bir hipotez (örneğin daha güçlü anizotropi, E4'ün kazanabileceği tasarımlar ya da farklı pilot boyutlarında veri verimliliği) ancak kendi önkaydıyla ele alınabilir ve bu koşunun başarısı olarak sunulamaz.

## Üç tür kanıt, birbirinden ayrı

**(a) Dondurulmuş bilimsel deney.** 29 dosya (kod, testler, `config.json`, `PROTOCOL.md`), bilimsel tohumla veri üretilmeden önce `FREEZE_MANIFEST.json` içinde hash'lendi. Çalıştırıcı, değişmiş kodu veya farklı Python/numpy/torch sürümünü reddeder. Seçimler, herhangi bir eğitim ya da test verisi üretilmeden önce hash'lendi (`selections.sha256`). Çıktılar `runs/scientific_01/` altında, denetim `analysis/` altında. Yukarıdaki bilimsel iddiaların tek kaynağı budur.

**(b) Mühendislik kontrolleri.** Birim testleri (59 adet), teknik smoke koşuları (`technical_smoke.json`, `runs_technical/smoke_02/`), çalışma süresi ölçümü (`technical_benchmark.json`) ve tasarım netleştirmelerinden önceki eski sürümler (`runs_technical/superseded_v1/`, `runs_technical/smoke_01/`) ayrı teknik tohumlar kullandı. Bunlar kodun iddia ettiği şeyi hesapladığını (Kronecker özdeşlikleri, gradyanlar, test verisinin yalıtımı, hash'leme, zamanlama) ve süre bütçesine sığdığını gösterir. Hipotezi sınamazlar; hipotez lehine ya da aleyhine bilimsel kanıt değildirler. Uygulayıcı, koruma kuralı verilmeden önce bazı çok erken deneme çıktılarını sildiğini açıkça bildirdi; bu yüzden geliştirme arşivi eksiksiz değildir (bkz. `IMPLEMENTATION_REPORT.md`).

**(c) Literatür ve yorum.** Özgünlük kontrolü, `RESULTS_TR.md` içindeki yorumlar ve üç yapay zekâ sohbet asistanının sonuç sonrası görüşleri (projede uygulama etiketleriyle kayıtlı: Claude Chat "Fable 5.1 Extra", ChatGPT "Pro", Antigravity "Gemini 4 Argon High") ölçüm değil, değerlendirmedir. Bu incelemeler aynı sonuç özetine dayanıyordu; ham dosyaların bağımsız denetimi değildi. Bir inceleyicinin ilk yanıtı uydurulmuş sayılar içeriyordu ve proje bunları açıkça reddetti. Karar oy sayısına değil, dondurulmuş eşiklere ve doğrulanmış kayıplara dayanır.

## Sınırlılıklar

- Yalnız sentetik ve doğrusal öğretmenler; doğrusal olmayan ağlar, dil modelleri, token yönlendirmesi veya yorumlanabilirlik hakkında hiçbir iddia yoktur.
- Hafif anizotropi (koşul sayısı 4). Güçlü anizotropi için bir sonuç çıkmaz.
- 12 bağımsız öğretmen bloğu; istatistiksel güç, kapsama veya anlamlılık iddiası yoktur.
- Referans, eğitilmiş en iyi sonlu adaydır; kanıtlanmış bir optimum değildir. Teknik tohum kontrolleri yaklaşık 0,0017 Z'ye kadar optimizasyon farkı buldu; bilimsel koşu izotropik popülasyon farkını en fazla 0,0072 Z olarak raporluyor (sonlu örnek ve optimizasyon birlikte). Bu yüzden çok küçük regret farklarına fazla anlam yüklenmemelidir.
- 64 boyutta 64 pilot örnekle gradyan tahmini gürültülüdür; yarım-pilot uyumu (24'te 16) bunu yansıtır.
- Kısa eğitim karşılaştırıcısı eşit tüketilen hesapla değil, eşit tanınan üst sınırla çalışır.
- E4 hiçbir ortamda kazanan olmadı.
- Adaylar arasında parametre sayısı eşittir; etkin hesap eşit değildir.

## Depo haritası

| Yol | İçerik |
|---|---|
| `soe/` | Çalışma kodu: Kronecker araçları, adaylar, ortam üreteci, seçiciler, kısa eğitim seçicisi, eğitim, değerlendirme, kaynak takibi, işlem hattı |
| `scripts/` | `run_pilot.py` (bilimsel çalıştırıcı), `freeze.py`, `technical_smoke.py`, `benchmark_runtime.py` |
| `tests/` | 59 birim testi (standart kütüphane `unittest`) |
| `config.json`, `PROTOCOL.md` | Dondurulmuş yapılandırma ve protokol (dondurma listesinde hash'li) |
| `FREEZE_MANIFEST.json` | 29 dondurulmuş dosyanın SHA-256 değerleri ve Python/numpy/torch sürümleri |
| `runs/scientific_01/` | Dondurulmuş bilimsel koşu: `summary.json`, `env_table.csv`, `candidate_seed_losses.csv`, `timing.csv`, `selections.json`, `records.json`, başlık ve manifest dosyaları |
| `runs/scientific_01/envs/` | Ortam başına 24 dizi dosyası (her biri yaklaşık 2,2 MB): öğretmen matrisleri, girdi kovaryansları, gürültü düzeyleri, eğitim/doğrulama/test yeterli istatistikleri, eğitim yörüngeleri ve dokuz küçük aday modelin iki başlangıç için eğitilmiş faktörleri ile yoğun matrisleri. Denetim bunları okur |
| `runs/scientific_01/pilot/` | 24 pilot dosyası (her biri yaklaşık 0,25 MB): görev başına ham 64 örneklik pilot girdi ve çıktıları; seçicilerin gördüğü tek veri |
| `analysis/` | `independent_audit.py` ve çıktısı `independent_audit.json`; `frozen_source.zip` (29 dondurulmuş dosya ve manifest arşivi) ve SHA-256 değeri |
| `technical_smoke.json`, `technical_benchmark.json`, `runs_technical/` | Teknik tohumlarla mühendislik kontrolleri (bilimsel kanıt değildir) |
| `RESULTS_TR.md` | Nihai sonuç ve karar raporu (Türkçe) |
| `IMPLEMENTATION_REPORT.md` | Uygulayıcı raporu: dosyalar, testler, tasarım itirazları, süreç açıklamaları |
| `docs/` | Uygulama talimatı, sonuç öncesi netleştirmeler, koşu öncesi denetim, İngilizce sonuç özeti |
| `PROJECT_NOTES.md` | Projenin orijinal README dosyası, değiştirilmeden (dahili çalışma notları; yorumlayıcı yolu makineye özgüdür) |

`PROTOCOL.md` hâlâ "PROVISIONAL — not frozen" başlığını taşır, çünkü dosya yazıldığı haliyle bayt bayt donduruldu. Bağlayıcılığını `FREEZE_MANIFEST.json` içindeki hash'i sağlar.

Bu açık kopyaya alınmayanlar: teknik smoke koşularının dizi dosyaları (`runs_technical/smoke_01/` ve `smoke_02/` altındaki `envs/*.npz` ve `pilot/*.npz`, yaklaşık 9,8 MB; bilimsel dizilerle aynı biçimde, yalnız teknik tohumlar, `technical_smoke.py` ile yeniden üretilebilir) dahili bir koordinasyon kaydı (`docs/coordination_log.md`; özel yapay zekâ sohbet oturumu bağlantıları içerir) ve orijinal öneri (`docs/original_proposal.txt`; projeye olduğu gibi yapıştırılmış Türkçe bir yapay zekâ asistanı yanıtı; fikri bu README'de özetlenmiş ve `PROTOCOL.md` içinde biçimselleştirilmiştir). Bu smoke koşularının `output_manifest.json` dosyaları dışarıda bırakılan dizileri hâlâ listeler; `RESULTS_TR.md` dışarıda bırakılan kayda bağlantı verir, `docs/implementation_brief.md` de iki dosyaya da atıf yapar.

## Yeniden üretme

Bağımlılıklar (`FREEZE_MANIFEST.json` ve `technical_benchmark.json` kayıtlarından): Python 3.11.13, numpy 2.4.6, torch 2.6.0 ve standart kütüphane (`unittest`). Başka paket kullanılmaz. Orijinal koşular macOS 26.6.2 arm64 (Apple M2 Max) üzerinde, CPU, float64 ve tek iş parçacığıyla yapıldı. Ayrı bir gereksinim dosyası yoktur. Bilimsel çalıştırıcı manifestteki Python, numpy veya torch sürümünden farklı bir sürümü reddettiği için tam olarak bu sürümleri içeren bir ortam kullanın.

Yollar: projenin kendi notları makineye özgü bir yorumlayıcı yolu kullanır (`/Users/emre/.../.venv-stage1a-mps-py311/bin/python`). Bunun yerine kendi `python` komutunuzu kullanın ve her şeyi deponun kök dizininden çalıştırın. Kod dosyaları depoya göre göreli olarak bulur (`scripts/_path.py`); düzenleme gerekmez.

Hiçbir eğitimi yeniden koşturmadan, yalnız eklenen dosyalarla doğrulanabilenler:

- `analysis/independent_audit.py` yalnız numpy gerektirir ve çalışma kodunu içe aktarmaz. `python analysis/independent_audit.py . runs/scientific_01` olarak çalıştırıldığında 29 dondurulmuş dosyayı koşu başlığındaki hash'lerle, 56 çıktı dosyasının tamamını (bütün `.npz` dosyaları dahil) `output_manifest.json` ile karşılaştırır; seçim hash'ini kontrol eder; her eğitilmiş adayın yoğun matrisini kayıtlı Kronecker faktörlerinden yeniden kurar; 432 test kaybını ve Z normalleştiricisini kayıtlı test istatistiklerinden yeniden hesaplar; ikinci ve üçüncü eşiği ve kararı beş görünümün hepsinde yeniden hesaplayıp `summary.json` ile karşılaştırır. Sonucu (`analysis/independent_audit.json` içeriğini) ekrana yazar, dosya yazmaz.
- Pilot dosyaları (`runs/scientific_01/pilot/envNN_pilot.npz`, üyeleri `X`, `Y`) seçicilerin gördüğü verinin aynısıdır. Bu nedenle `selections.json` içindeki seçici skorları ve seçimleri `PROTOCOL.md` formülleriyle bağımsız olarak yeniden hesaplanabilir; bunun için hazır bir betik eklenmemiştir.
- Kısa eğitim karşılaştırıcısının seçimleri dosyalardan birebir yeniden hesaplanamaz, çünkü bir saniyeye kaç tur sığdığına bağlıdır; kullanılan tur sayıları `selections.json` içinde kayıtlıdır.

Sıfırdan yeniden koşturma:

```bash
# birim testleri (yaklaşık 5 s; MPS testi yalnız Apple MPS bulunan makinede çalışır)
python -m unittest discover -s tests -t . -v

# teknik tohumlarla mühendislik kontrolleri; çıktılar asla üzerine yazılmaz, yeni adlar kullanın
python scripts/benchmark_runtime.py --out technical_benchmark_rerun.json
python scripts/technical_smoke.py --out technical_smoke_rerun.json --run-dir runs_technical/smoke_rerun

# dondurulmuş bilimsel deneyi YENİ bir dizine yeniden koşturma (çalıştırıcı var olan dizini reddeder)
python scripts/run_pilot.py --freeze FREEZE_MANIFEST.json --out runs/scientific_rerun

# eklenen koşunun ya da yeni bir koşu dizininin bağımsız denetimi
python analysis/independent_audit.py . runs/scientific_01
python analysis/independent_audit.py . runs/scientific_rerun
```

Notlar:

- `run_pilot.py`, `soe/`, `scripts/` veya `tests/` altında değişmiş, eksik ya da fazladan `.py` dosyasını ve `config.json` veya `PROTOCOL.md` üzerindeki herhangi bir değişikliği reddeder. Orijinal dondurma listesini kullanmak istiyorsanız bu dizinlerde dosya değiştirmeyin ve eklemeyin.
- Kısa eğitim karşılaştırıcısı duvar saati süresine bağlıdır; başka donanımda tur sayıları, hatta seçimleri biraz farklı çıkabilir. Bütün süreler farklı olacaktır.
- Sıfırdan bir koşu aynı tohumları kullanır; öğretmenler, veriler ve analitik seçimler yeniden üretilmelidir. Ancak farklı donanım veya BLAS derlemelerinde kayan nokta sonuçlarının bit düzeyinde aynı çıkması garanti değildir; kayıtlı kanıt, eklenen dizilerdir.
- Bazı kayıtlı dosyalar orijinal makinenin mutlak yollarını (`/Users/emre/...`) içerir: `runs/scientific_01/run_header.json` (dondurma listesinin yolu), `IMPLEMENTATION_REPORT.md`, `docs/implementation_brief.md` ve `PROJECT_NOTES.md`. Bunlar kanıt olarak değiştirilmeden tutuldu; `run_header.json` ayrıca `output_manifest.json` hash'leri kapsamındadır, düzenlenirse denetim başarısız olur. Bu yollar yeniden koşturmayı etkilemez.

## Kaynaklar

Projenin kendi belgelerinde atıf yapılan kaynaklar. İlk dördü proje tarafından kontrol edildi (başlıklar, özetler ve ilgili bölümler; kapsamlı bir özgünlük taraması değildir). Diğerleri orijinal öneride (`docs/original_proposal.txt`, bu kopyada yok) geçer ve ayrıca yeniden doğrulanmamıştır.

Proje tarafından kontrol edilenler:

- TensorPoly: Mixture of Latent Experts Using Tensor Products (2024). https://arxiv.org/abs/2405.16671
- MoKA (COLING 2025), Kronecker tabanlı seyrek uzman karışımı. https://aclanthology.org/2025.coling-main.679/
- CDKA, Diving into Kronecker Adapters: Component Design Matters (ICML 2026). https://arxiv.org/abs/2602.01267 (kontrol edilen sürüm: https://arxiv.org/html/2602.01267v3)
- DR-LoRA (COLM 2026), yönlendirme ve gradyan önemine göre uzman LoRA rank dağıtımı. https://arxiv.org/abs/2601.04823 (kontrol edilen sürüm: https://arxiv.org/html/2601.04823v6)

Yalnız orijinal öneride geçenler:

- MoKA: Mixture of Kronecker Adapters (2025). https://arxiv.org/abs/2508.03527
- MAdaKron (2026). https://www.sciencedirect.com/science/article/abs/pii/S0950705125021240
- Scaling Laws for Fine-Grained Mixture of Experts (ICML 2024). https://proceedings.mlr.press/v235/ludziejewski24a.html
- Standart en iyi Kronecker yaklaşımı sonucu için atıf yapılan Cornell eCommons kaydı. https://ecommons.cornell.edu/entities/publication/ad6a5c71-2632-4652-b33a-72016d7abc4d
- Tensör ayrıştırılabilirliği ile ağların ifade kapasitesi arasındaki bağ için atıf yapılan arXiv makalesi. https://arxiv.org/abs/1704.01552
- Dense ve MoE modellerde tensör parametrizasyonlarının sınırları için atıf yapılan ACL Anthology makalesi. https://aclanthology.org/2026.emnlp-main.2679/
- GLUE benchmark (Stage B için önerilen veri). https://arxiv.org/abs/1804.07461
- Qwen2.5-0.5B-Instruct yapılandırması (Stage B için önerilen model). https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json

## Köken ve lisans

- Çalışma, depo sahibinin isteği ve yetkilendirmesiyle yürütüldü.
- Uygulama kodu ve testler, `IMPLEMENTATION_REPORT.md` ve `RESULTS_TR.md` dosyalarında belirtildiği üzere Claude Code (uygulama etiketi "Opus 5.5 High") tarafından yazıldı.
- Ayrı bir koordinatör yapay zekâ ajanı kaynağı denetledi, dondurdu, bilimsel deneyi koşturdu ve bağımsız denetimi yazdı; dondurma listesinde onaylayan olarak "Codex coordinator: user-authorized pilot" kayıtlıdır. Proje kayıtlarına göre koordinatör bilimsel uygulama veya test kodunu yazmadı ya da değiştirmedi; dondurmadan önce `PROTOCOL.md` içinde yakınsamayla ilgili iki ifadeyi düzeltti.
- Tasarım koşudan önce, sonuçlar da koşudan sonra üç yapay zekâ sohbet asistanı tarafından eleştirildi (bkz. "Üç tür kanıt" bölümü).
- Bütün deneyler 31 Ekim 2026'da tek bir Apple M2 Max makinede yerel olarak çalıştırıldı.

Lisans dosyası eklenmemiştir; aksi belirtilmedikçe bütün haklar depo sahibine aittir. Yeniden kullanmadan önce depo sahibiyle iletişime geçin.
