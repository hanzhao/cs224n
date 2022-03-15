// Deno
const THRESHOLD = 0.95;

async function readCsv(file: string, column: number): Promise<string[][]> {
  const text = await Deno.readTextFile(file);
  let header = true;
  const results: string[][] = [];
  for (const line of text.split('\n')) {
    if (header || !line) {
      header = false;
      continue;
    }
    const cells = line.split(',');
    const result = [];
    for (let i = 0; i < column; ++i) {
      result.push(cells[i]);
    }
    for (let i = column; i < cells.length; ++i) {
      result[column - 1] += ',' + cells[i];
    }
    results.push(result);
  }
  return results;
}

async function gao(gatePath: string, expertOutputPaths: string[], savePath: string): Promise<void> {
  const gate = await readCsv(gatePath, 3);
  const expertOutputs = await Promise.all(expertOutputPaths.map(e => readCsv(e, 2)));
  for (const expertOutput of expertOutputs) {
    if (expertOutput.length !== gate.length) {
      console.error(`Line counts of files don't match for expert! ${expertOutput.length} vs ${gate.length}`);
      return;
    }
  }
  let text = 'Id,Predicted\n';
  for (let i = 0; i < gate.length; ++i) {
    for (let j = 0; j < expertOutputs.length; ++j) {
      if (expertOutputs[j][i][0] !== gate[i][0]) {
        console.error(`IDs don't match for expert! ${expertOutputs[j][i][0]} vs ${gate[i][0]}`);
        return;
      }
    }
    const source = Number.parseInt(gate[i][1], 10);
    const score = Number.parseFloat(gate[i][2]);
    let output = '';
    if (score < THRESHOLD) {
      output = expertOutputs[0][i][1];
    } else {
      output = expertOutputs[source][i][1];
    }
    text += `${gate[i][0]},${output}\n`;
  }
  await Deno.writeTextFile(savePath, text, {create: true});
}

async function main() {
  await gao('save/moe-stage/validation_submission.csv', [
    'save/finetune-all-stage/validation_submission.csv',
    'save/finetune-duorc-stage/validation_submission.csv',
    'save/finetune-race-stage/validation_submission.csv',
    'save/finetune-re-stage/validation_submission.csv',
  ], 'save/moe-validation/submission.csv');
  await gao('save/moe-stage/test_submission.csv', [
    'save/finetune-all-stage/test_submission.csv',
    'save/finetune-duorc-stage/test_submission.csv',
    'save/finetune-race-stage/test_submission.csv',
    'save/finetune-re-stage/test_submission.csv',
  ], 'save/moe-test/submission.csv');
}

main()