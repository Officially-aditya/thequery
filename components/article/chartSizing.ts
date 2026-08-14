const MIN_HORIZONTAL_BAR_CHART_HEIGHT = 180;
const BAR_ROW_HEIGHT = 36;
const CHART_CHROME_HEIGHT = 56;

/**
 * Give horizontal bar charts enough room for each category while keeping
 * short charts from becoming oversized in the article rail.
 */
export function getHorizontalBarChartHeight(rowCount: number): number {
  return Math.max(
    MIN_HORIZONTAL_BAR_CHART_HEIGHT,
    rowCount * BAR_ROW_HEIGHT + CHART_CHROME_HEIGHT,
  );
}
