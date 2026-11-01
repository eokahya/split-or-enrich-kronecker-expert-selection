# Split or Enrich — önkayıtlı pilot sonucu

31 Ekim 2026. Ana deney tamamlandı. Sonuç: **NO_ADDED_UTILITY**. Bu, güçlü karşılaştırıcıya göre önceden istenen ek doğruluk faydasının gösterilemediği anlamına gelir. Adaptif mimari seçiminin faydasız olduğu veya Kronecker–MoE ilişkisinin çürütüldüğü anlamına gelmez.

## Ne yaptık?

Fikir, gerçek uygulama oturumlarında Claude Chat **Fable 5.1 Extra**, Safari **GPT Pro** ve Antigravity **Gemini 4 Argon High** arasında eleştirildi. Görüşler karşılıklı aktarıldı; matematik hataları, desteklenmeyen karşıörnekler ve uygun olmayan başarı ölçütleri sorgulandı. Üçü de başlangıçta yalnız sınırlı sentetik pilota onay verdi. Bu etiketler uygulama arayüzünde gözlenen model adlarıdır.

Kod ve testleri Claude Code **Opus 5.5 High** hazırladı. Koordinatör kaynak denetimini ve bağımsız test çalıştırmasını yaptı; bilimsel uygulama/test kodunu değiştirmedi. Protokolde yakınsama hakkında iki aşırı güçlü ifadeyi düzeltti. 29 dosya bilimsel veri açılmadan hash ile sabitlendi.

M2 Max üzerinde CPU/float64 ve tek iş parçacığıyla **24 ortam × 9 aday × 2 başlangıç = 432 eğitim**, her birinde 400 Adam adımı çalıştırıldı. Toplam süre testler ve ısınma dahil **112,50 saniye**. **59 test geçti**, atlanan test yok; MPS doğruluk testi de çalıştı. Bu, ana bilimsel eğitimin GPU üzerinde yapıldığı anlamına gelmez.

Dört görev, 64 boyut, eşit 1.024 faktör parametresi: E1/r8, yedi E2/r4 bölümlenmesi, E4/r2. Öğretmen karmaşıklığı ve görev farklılığı için dört hücre, hücre başına üç bağımsız öğretmen kullanıldı. Her öğretmen izotropik ve koşul sayısı 4 olan anizotropik girdilerle eşlendi: **12 bağımsız blok**, 24 bağımsız tekrar değil. Görev başına pilot/eğitim/doğrulama/test örnek sayıları 64/512/128/2048.

## Önceden konan başarı koşulları ve sonuç

Normalize regret, seçilen adayın test kaybı ile aynı eğitim bütçesiyle eğitilen en iyi adayın kaybı arasındaki farkın, gürültü dahil beklenen sıfır-model kaybına bölünmesidir. Referans küresel optimum değildir.

| Koşul | Önceden belirlenen eşik | Ölçüm | Karar |
|---|---|---|---|
| Teknik geçerlilik ve tamamlama | Tüm testler/nümerik kontroller; tüm 24 ortam; ≤3.600 s | Geçti; 112,50 s | Geçti |
| İyi seçim | En az 20/24 ortamda regret ≤0,02 | 24/24; en kötü 0,0030454 | Geçti |
| Ek doğruluk faydası | Anizotropik ortalamada en güçlü karşılaştırıcıdan ≥0,01 daha iyi | Naif 0,0003992; en güçlü 0,0003804 | Kaldı |

Son koşulda karşılaştırıcılar kovaryanslı skor, kısa eğitimle seçim ve dokuz sabit adaydır. Önce her karşılaştırıcının aynı 12 ortamdaki ortalaması hesaplanır; sonra bu ortalamaların en iyisi alınır. Her ortam için farklı bir sabit aday seçilmez.

| Yöntem | Anizotropik ortalama regret / Z | Z'nin yüzdesi | Medyan seçim süresi |
|---|---:|---:|---:|
| Birincil naif Kronecker skoru | 0,000399225 | %0,0399 | 3,660 ms |
| Kovaryanslı skor | 0,000380419 | %0,0380 | 4,591 ms |
| Kısa eğitimle seçim | 0,000957333 | %0,0957 | 1.002,067 ms |
| İkincil crossfit | 0,001593191 | %0,1593 | 17,965 ms |
| En iyi sabit aday: E1/r8 | 0,021003398 | %2,1003 | Sabit seçim |

Kısa eğitim karşılaştırıcısına **dokuz adayın tamamı için toplam 1 saniyelik ortak üst bütçe** verildi; yöntem başına gerçekten tüketilen süreler eşit değildir. Yalnız tamamlanan son bütçe-içi tur kullanıldı; raporlanan süre, kullanılmayan son taşma turunu da içerir. Naif yöntemle gerçekten eş süreli ikincil eğitim denemesi 24/24 ortamda ölçülemedi; ona yapay kötü skor verilmedi.

Kovaryanslı karşılaştırıcı 0,01'den çok daha düşük regret'e sahip olduğu için, bu veri üzerinde kusursuz bir seçici bile 0,01 üstünlük eşiğini geçemez. Eşiği sonuçtan sonra değiştirmedik. Beş duyarlılık görünümü — iki başlangıcın ortalaması, başlangıç 0, başlangıç 1, aday başına minimum ve tam popülasyon riski — aynı **NO_ADDED_UTILITY** kararını verdi.

## Güçlü ve zayıf yönler

**Güçlü:** “Farklı işler için daha çok uzman; karmaşık ortak iş için daha fazla uzman kapasitesi” ayrımı anlaşılır ve izotropik doğrusal modelde matematiksel olarak açık. Ölçülen seçici, sabit E1/r8'e göre anizotropik ortalamada yaklaşık **0,020604 Z** kazanç sağladı. Ucuz pilot ölçümünün bu sınırlı problem ailesinde kötü kapasite tercihlerinden kaçınabildiği gösterildi. Hafif anizotropide çökme beklentisi bu deneyle desteklenmedi.

**Zayıf:** Kovaryans bilgisini kullanan basit rakibe ek katkı yok. Yeni teorem, özgün MI mekanizması veya LLM aktarımı gösterilmedi. Başlangıç gradyanı hedef operatörü ve girdi kovaryansını birlikte taşır; tek başına genel optimumu belirlemez. Kronecker/MoE birleşimi ve başlangıç gradyanı geometrisi yakın literatürde zaten vardır.

E1 12, E2 12 ortamda kazandı; **E4 hiçbirinde kazanmadı**. Dolayısıyla üç kapasite rejiminin tamamı sınanmış sayılmaz. Yarım-pilot seçimleri 16/24 kez uyuştu, ortalama sıralama korelasyonu 0,654 idi. Düşük regret, tam kazananı her seferinde doğru saptamakla aynı değildir. İki başlangıcın kazananları 21/24 ortamda uyuştu. 432 eğitimin ikisinde son 50 adımlık kayıp azalması tanı eşiğini aştı; bu tanı sonuçtan sonra kaçış kapısı yapılmadı. Küçük naif–kovaryans farkı ne istatistiksel üstünlük ne eşdeğerlik kanıtıdır. Popülasyon optimumundan uzaklık, bu fark için bir güven aralığı değildir.

## Karar ve kayıt

Koordinatör kararı: **Bu pilotu mevcut iddiasıyla kapat; büyük LLM aşamasını başlatma.** Ölçülen adaptif kazancı teknik sonuç olarak koru. Güçlü karşılaştırıcının sağlamadığı fayda ve yenilik ortaya konmadan daha büyük deneye yatırım yapma. Ayrı yeni bir hipotez ancak ayrı önkayıtla değerlendirilebilir; bu koşunun başarısı olarak sunulamaz.

Üç model de gerçek sonuçları değerlendirdikten sonra büyük LLM aşamasına **NO-GO** dedi. Gerekçeleri ve olası ayrı devam önerileri aynı değil:

- **GPT Pro:** Hafif anizotropide bozulma beklentisini veri karşısında zayıflattı. Adaptif kazancı savundu; kendi önerdiği üstünlük eşiğinin bu sonuçta geçilemez hale geldiğini kabul etti. Olası ayrı çalışma: taze öğretmenlerde 16/32/64 pilot örnekle veri verimliliği.
- **Fable Extra:** Beklediği pilot gürültüsünün regret'i belirgin bozmadığını kabul etti; E4 rejiminin eksikliğini ve sıralama kararsızlığını vurguladı. Olası ayrı çalışma: daha güçlü anizotropi, farklı örnek sayıları ve E4'ün kazanabildiği önceden tasarlanmış öğretmenler. “Eşdeğer yöntemler” ve “optimizasyon gürültü tabanı” ifadelerini kanıtlanmış sonuç olarak benimsemiyoruz.
- **Antigravity:** İlk yorumundaki hayali E4 kazançlarını ve yanlış süreleri, gerçek E4=0 ve ölçülen süreler geldikten sonra değiştirdi. Son kararında doğrudan Qwen'e geçmeyi reddetti. Olası ayrı çalışma: doğrusal olmayan sınıflandırma kaybıyla bir ara deney. Uydurulmuş exact-hit/oracle/permütasyon kontrolü sayıları rapora alınmadı. Öğretmenleri sonuca bakıp belirli kazanan dağılımını tutturana kadar seçme önerisini de benimsemiyoruz.

Bu sonuçlar üç modelin bağımsız ham dosya denetimi değildir; aynı ölçüm özeti üzerinden gerçek uygulama yanıtlarıdır. Nihai karar, oy sayısına değil dondurulmuş ölçütlere ve doğrulanmış kayıplara dayanır. Yeni deney başlatılmadı.

Kaynak ve çıktı hashleri doğrulandı. Koordinatör, araştırma modüllerini kullanmadan faktörlerden matrisleri, kayıpları, normalizeri ve beş karar görünümünü yeniden hesapladı; en büyük normalize kayıp farkı 7,01×10⁻¹⁶. Ayrı teknik tohumlarla geliştirme yapıldığı açıktır. Opus, ilk teknik probe dosyalarını koruma talimatından önce sildiğini bildirdi; bu nedenle eksiksiz geliştirme arşivi iddiası yok. Sonraki v1/v2 ve bütün bilimsel çıktılar korunuyor.

Yakın literatür: [MoKA — Kronecker uzmanları](https://aclanthology.org/2025.coling-main.679/), [CDKA — bileşen tasarımı ve başlangıç gradyanı geometrisi](https://arxiv.org/html/2602.01267v3), [DR-LoRA — gradyan/routing ile rank dağıtımı](https://arxiv.org/html/2601.04823v6). Bunlar dar bir öncül kontrolüdür, kapsamlı özgünlük garantisi değildir.

Dosyalar: [protokol](PROTOCOL.md), [sabitlenen kaynak listesi](FREEZE_MANIFEST.json), [dondurulmuş kaynak arşivi](analysis/frozen_source.zip), [ana sonuç](runs/scientific_01/summary.json), [ortam tablosu](runs/scientific_01/env_table.csv), [432 eğitim kaydı](runs/scientific_01/candidate_seed_losses.csv), [bağımsız denetim](analysis/independent_audit.json), [gerçek model tartışma kaydı](docs/coordination_log.md), [Opus uygulama raporu](IMPLEMENTATION_REPORT.md).
