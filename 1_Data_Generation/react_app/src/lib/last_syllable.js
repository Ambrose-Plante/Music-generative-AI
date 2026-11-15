function findLastSyllable(arr) {
  let ind = 0;

  for (let i = 0, len=arr.length; i < len; i++) {
    let v = arr[i].syllable;
    min = (v < min) ? v : min;
    max = (v > max) ? v : max;
  }

  return [min, max];
}

export default findMinMax
