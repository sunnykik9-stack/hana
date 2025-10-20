async function compute() {
  const a = {
    name: document.getElementById('a_name').value || 'A',
    date: document.getElementById('a_date').value || '1990-05-10',
    time: document.getElementById('a_time').value || '08:20',
    tz:   document.getElementById('a_tz').value   || 'Asia/Seoul',
  };
  const b = {
    name: document.getElementById('b_name').value || 'B',
    date: document.getElementById('b_date').value || '1992-11-22',
    time: document.getElementById('b_time').value || '23:45',
    tz:   document.getElementById('b_tz').value   || 'Asia/Seoul',
  };

  const btn = document.getElementById('btn');
  btn.disabled = true;
  btn.innerText = '계산 중...';

  try {
    const res = await fetch('/compute-synastry', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ personA: a, personB: b })
    });
    const data = await res.json();
    document.getElementById('result').style.display = 'block';
    document.getElementById('score').innerText = `스코어: ${data.score}`;
    const ul = document.getElementById('aspects');
    ul.innerHTML = '';
    (data.aspects_top || []).forEach(x => {
      const li = document.createElement('li');
      li.textContent = `${x.bodies.join(' × ')} — ${x.type} (orb ${x.orb}) · ${x.score_contrib >= 0 ? '+' : ''}${x.score_contrib}`;
      ul.appendChild(li);
    });
    document.getElementById('summary').innerText = data.summary;
  } catch (e) {
    alert('에러: ' + e);
  } finally {
    btn.disabled = false;
    btn.innerText = '궁합 계산하기';
  }
}

document.getElementById('btn').addEventListener('click', compute);
