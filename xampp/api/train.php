<?php
declare(strict_types=1);

// SSE-Setup: Output-Puffer leeren und Streaming aktivieren
set_time_limit(300);
ini_set('output_buffering', 'off');
ini_set('zlib.output_compression', 'false');
ini_set('implicit_flush', '1');

if (function_exists('apache_setenv')) {
    apache_setenv('no-gzip', '1');
}
while (ob_get_level() > 0) {
    ob_end_clean();
}
ob_implicit_flush(true);

header('Content-Type: text/event-stream; charset=utf-8');
header('Cache-Control: no-cache, no-store');
header('X-Accel-Buffering: no');
header('Access-Control-Allow-Origin: *');
header('Connection: keep-alive');

require_once __DIR__ . '/../lib/StockData.php';
require_once __DIR__ . '/../lib/GeneticClassifier.php';

function sse(array $payload): void
{
    echo 'data: ' . json_encode($payload, JSON_UNESCAPED_UNICODE) . "\n\n";
    flush();
}

function sseFehler(string $msg): never
{
    sse(['fehler' => $msg]);
    exit;
}

// Parameter einlesen
$ticker      = strtoupper(trim($_GET['ticker']      ?? 'AAPL'));
$period      = $_GET['period']      ?? '1y';
$generations = min((int) ($_GET['generations'] ?? 30), 100);
$population  = min((int) ($_GET['population']  ?? 50), 200);

// Erste SSE-Nachricht sofort senden (verhindert Browser-Timeout beim Datenabruf)
sse(['status' => 'laden', 'nachricht' => 'Marktdaten werden abgerufen…']);

// Daten von Yahoo Finance holen
try {
    $result = buildFeatures($ticker, $period);
} catch (Exception $e) {
    sseFehler($e->getMessage());
}

$rows = $result['rows'];
if (count($rows) < 30) {
    sseFehler('Zu wenig Daten für das Training (mindestens 30 Handelstage benötigt).');
}

sse(['status' => 'training', 'nachricht' => 'Evolution gestartet…']);

[$trainRows, $testRows] = trainTestSplit($rows);

$XTrain = array_column($trainRows, 'features');
$yTrain = array_column($trainRows, 'target');
$XTest  = array_column($testRows,  'features');
$yTest  = array_column($testRows,  'target');

// Modell trainieren – Fortschritt per SSE streamen
$clf = new GeneticRuleClassifier($population, $generations);

$clf->fit(
    $XTrain,
    $yTrain,
    FEATURE_NAMES,
    function (int $gen, int $total, float $score): void {
        sse(['gen' => $gen, 'total' => $total, 'score' => round($score, 4)]);
    }
);

// Genauigkeit berechnen
$trainAcc = calcAccuracy($clf->predict($XTrain), $yTrain);
$testAcc  = calcAccuracy($clf->predict($XTest),  $yTest);

// Vorhersage auf den letzten 30 Handelstagen
$last30   = array_slice($rows, -30);
$X30      = array_column($last30, 'features');
$y30      = array_values(array_column($last30, 'target'));
$preds30  = array_values($clf->predict($X30));
$lastPred = end($preds30);

// Abschlussereignis mit allen Ergebnissen
sse([
    'fertig'                 => true,
    'train_genauigkeit'      => round($trainAcc, 4),
    'test_genauigkeit'       => round($testAcc,  4),
    'fitness_verlauf'        => $clf->history,
    'regeln'                 => $clf->getRules(),
    'anzahl_regeln'          => count($clf->bestModel->rules),
    'daten_pred'             => array_column($last30, 'date'),
    'kurse_pred'             => array_column($last30, 'close'),
    'vorhersagen'            => $preds30,
    'tatsaechlich'           => $y30,
    'letzte_vorhersage'      => (int) $lastPred,
    'letzte_vorhersage_text' => $lastPred === 1 ? 'Steigt ↑' : 'Fällt ↓',
]);
