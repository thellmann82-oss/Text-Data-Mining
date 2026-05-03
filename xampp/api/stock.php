<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=utf-8');
header('Access-Control-Allow-Origin: *');

require_once __DIR__ . '/../lib/StockData.php';

$ticker = strtoupper(trim($_GET['ticker'] ?? ''));
$period = $_GET['period'] ?? '1y';

if ($ticker === '') {
    http_response_code(400);
    echo json_encode(['fehler' => 'Kein Ticker angegeben.']);
    exit;
}

try {
    $result = buildFeatures($ticker, $period);
    $rows   = $result['rows'];
    $n      = count($rows);

    if ($n === 0) {
        throw new RuntimeException('Keine verwertbaren Daten gefunden.');
    }

    $upCount = array_sum(array_column($rows, 'target'));

    echo json_encode([
        'ticker'    => $result['ticker'],
        'name'      => $result['name'],
        'currency'  => $result['currency'],
        'sektor'    => '',
        'anzahl'    => $n,
        'daten'     => array_column($rows, 'date'),
        'kurse'     => array_column($rows, 'close'),
        'up_anteil' => round($upCount / $n * 100.0, 1),
        'merkmale'  => FEATURE_NAMES,
    ]);
} catch (Exception $e) {
    http_response_code(500);
    echo json_encode(['fehler' => $e->getMessage()]);
}
