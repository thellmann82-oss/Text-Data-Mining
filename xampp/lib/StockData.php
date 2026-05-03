<?php
declare(strict_types=1);

const FEATURE_NAMES = [
    'Tagesrendite',
    'Rendite (3T)',
    'Rendite (5T)',
    'MA5-Verhältnis',
    'MA10-Verhältnis',
    'MA20-Verhältnis',
    'Volumenänderung',
    'Hoch-Tief-Spanne',
    'Oberer Docht',
    'Unterer Docht',
];

/**
 * Holt OHLCV-Daten von Yahoo Finance und berechnet technische Merkmale.
 */
function buildFeatures(string $ticker, string $period): array
{
    $result = fetchYahoo($ticker, $period);

    $timestamps = $result['timestamp']                          ?? [];
    $quote      = $result['indicators']['quote'][0]             ?? [];
    $open       = $quote['open']   ?? [];
    $high       = $quote['high']   ?? [];
    $low        = $quote['low']    ?? [];
    $close      = $quote['close']  ?? [];
    $volume     = $quote['volume'] ?? [];
    $meta       = $result['meta']  ?? [];
    $m          = count($close);

    // Filtere ungültige Zeilen (null-Werte)
    $valid = [];
    for ($i = 0; $i < $m; $i++) {
        if (isset($close[$i], $open[$i], $high[$i], $low[$i], $volume[$i])
            && $close[$i] !== null && $volume[$i] !== null) {
            $valid[] = $i;
        }
    }

    $cls  = array_map(fn($i) => (float) $close[$i],   $valid);
    $hig  = array_map(fn($i) => (float) $high[$i],    $valid);
    $low_ = array_map(fn($i) => (float) $low[$i],     $valid);
    $opn  = array_map(fn($i) => (float) $open[$i],    $valid);
    $vol  = array_map(fn($i) => (float) $volume[$i],  $valid);
    $ts   = array_map(fn($i) => (int)   $timestamps[$i], $valid);
    $n    = count($cls);

    // Gleitende Durchschnitte
    $ma = function (array $arr, int $w) use ($n): array {
        $out = array_fill(0, $n, null);
        for ($i = $w - 1; $i < $n; $i++) {
            $sum = 0.0;
            for ($j = $i - $w + 1; $j <= $i; $j++) $sum += $arr[$j];
            $out[$i] = $sum / $w;
        }
        return $out;
    };

    $ma5  = $ma($cls, 5);
    $ma10 = $ma($cls, 10);
    $ma20 = $ma($cls, 20);

    $rows = [];
    for ($i = 21; $i < $n - 1; $i++) {
        // Renditen
        $ret1 = $cls[$i - 1] != 0.0 ? ($cls[$i] - $cls[$i - 1]) / $cls[$i - 1] : null;
        $ret3 = $cls[$i - 3] != 0.0 ? ($cls[$i] - $cls[$i - 3]) / $cls[$i - 3] : null;
        $ret5 = $cls[$i - 5] != 0.0 ? ($cls[$i] - $cls[$i - 5]) / $cls[$i - 5] : null;

        $ma5r  = $ma5[$i]  !== null && $ma5[$i]  != 0.0 ? $cls[$i] / $ma5[$i]  : null;
        $ma10r = $ma10[$i] !== null && $ma10[$i] != 0.0 ? $cls[$i] / $ma10[$i] : null;
        $ma20r = $ma20[$i] !== null && $ma20[$i] != 0.0 ? $cls[$i] / $ma20[$i] : null;

        $volChg = $vol[$i - 1] != 0.0 ? ($vol[$i] - $vol[$i - 1]) / $vol[$i - 1] : null;

        $hl       = $hig[$i] - $low_[$i];
        $hlRatio  = $hl > 0.0 ? $hl / $cls[$i] : 0.0;
        $bodyHi   = max($opn[$i], $cls[$i]);
        $bodyLo   = min($opn[$i], $cls[$i]);
        $upWick   = $hl > 0.0 ? ($hig[$i] - $bodyHi) / $hl : 0.0;
        $loWick   = $hl > 0.0 ? ($bodyLo - $low_[$i]) / $hl : 0.0;

        if ($ret1 === null || $ret3 === null || $ret5 === null
            || $ma5r === null || $ma10r === null || $ma20r === null
            || $volChg === null) {
            continue;
        }

        $rows[] = [
            'features' => [$ret1, $ret3, $ret5, $ma5r, $ma10r, $ma20r, $volChg, $hlRatio, $upWick, $loWick],
            'target'   => $cls[$i + 1] > $cls[$i] ? 1 : 0,
            'date'     => date('Y-m-d', $ts[$i]),
            'close'    => round($cls[$i], 2),
        ];
    }

    return [
        'rows'     => $rows,
        'meta'     => $meta,
        'ticker'   => strtoupper($ticker),
        'currency' => $meta['currency'] ?? 'USD',
        'name'     => $meta['longName'] ?? $meta['shortName'] ?? strtoupper($ticker),
    ];
}

function fetchYahoo(string $ticker, string $period): array
{
    $url = sprintf(
        'https://query2.finance.yahoo.com/v8/finance/chart/%s?interval=1d&range=%s',
        urlencode(strtoupper($ticker)),
        urlencode($period)
    );

    $ch = curl_init($url);
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT        => 30,
        CURLOPT_FOLLOWLOCATION => true,
        CURLOPT_SSL_VERIFYPEER => false,
        CURLOPT_HTTPHEADER     => [
            'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept: application/json',
            'Accept-Language: de-DE,de;q=0.9',
        ],
    ]);

    $body     = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $curlErr  = curl_error($ch);
    curl_close($ch);

    if ($body === false || $body === '') {
        throw new RuntimeException('Netzwerkfehler: ' . $curlErr);
    }
    if ($httpCode !== 200) {
        throw new RuntimeException("Yahoo Finance HTTP-Fehler: $httpCode");
    }

    $data = json_decode($body, true);
    if (!isset($data['chart']['result'][0])) {
        $errMsg = $data['chart']['error']['description'] ?? 'Ticker nicht gefunden';
        throw new RuntimeException("Yahoo Finance: $errMsg");
    }

    return $data['chart']['result'][0];
}

function trainTestSplit(array $rows, float $testSize = 0.2): array
{
    $split = (int) floor(count($rows) * (1.0 - $testSize));
    return [array_slice($rows, 0, $split), array_slice($rows, $split)];
}

function calcAccuracy(array $preds, array $y): float
{
    $n = count($y);
    if ($n === 0) return 0.0;
    $correct = 0;
    for ($i = 0; $i < $n; $i++) {
        if ($preds[$i] === $y[$i]) $correct++;
    }
    return $correct / $n;
}
