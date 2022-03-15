// Deno

async function gao(input: string, output: string): Promise<void> {
  const data = JSON.parse(await Deno.readTextFile(input));
  await Deno.writeTextFile(output, JSON.stringify({
    "data": data.data.slice(0, 128),
  }), {create: true});
}

async function main(): Promise<void> {
  await gao('datasets/indomain_train/nat_questions', 'datasets/indomain_sample/nat_questions');
  await gao('datasets/indomain_train/newsqa', 'datasets/indomain_sample/newsqa');
  await gao('datasets/indomain_train/squad', 'datasets/indomain_sample/squad');
}

main()